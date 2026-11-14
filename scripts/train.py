"""Train StereoFlowFormer.

  python scripts/train.py --config configs/vkitti2_sup.yaml            # stage 1
  python scripts/train.py --config configs/kitti_selfsup.yaml          # stage 2
  python scripts/train.py --config ... --set train.iters=200 train.val_every=100   # overrides
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvo.data.kitti import KITTIOdometry  # noqa: E402
from dvo.data.vkitti2 import SCENES as VKITTI2_SCENES, VKITTI2  # noqa: E402
from dvo.eval.metrics import disp_metrics, flow_metrics, moving_iou, pose_metrics  # noqa: E402
from dvo.geometry.warp import fb_consistency  # noqa: E402
from dvo.losses.motion_seg import motion_seg_loss  # noqa: E402
from dvo.losses.photometric import flow_photometric_loss, smoothness, stereo_photometric_loss  # noqa: E402
from dvo.losses.pose import pose_kabsch_consistency, pose_rigid_flow_loss, pose_supervised_loss  # noqa: E402
from dvo.losses.supervised import sequence_loss  # noqa: E402
from dvo.models.model import StereoFlowFormer  # noqa: E402
from dvo.utils import disp_to_color, flow_to_color, load_config, masks_to_color, to_np  # noqa: E402


def infinite(loader):
    while True:
        for b in loader:
            yield b


def make_loader(ds, bs, workers, shuffle=True):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=workers, pin_memory=True, drop_last=shuffle,
                      persistent_workers=workers > 0, worker_init_fn=getattr(ds, 'worker_init', None))


def to_cuda(batch):
    return {k: v.cuda(non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}


def forward(model, b):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(b['Lt'], b['Rt'], b['Lt1'], b['Rt1'], b['K'], b['baseline'])


def compute_losses(out, b, cfg, it, supervised, frozen=False):
    L = cfg['loss']
    terms, stats = {}, {}
    B = b['Lt'].shape[0]
    flow_fw, flow_bw = out['flow_fw'].float(), out['flow_bw'].float()
    occ_fw = fb_consistency(flow_fw.detach(), flow_bw.detach())
    occ4 = (F.avg_pool2d(occ_fw, 4) > 0.5).float()

    if supervised and not frozen:
        gt_flow = torch.cat([b['flow_fw'], b['flow_bw']])
        gt_fv = torch.cat([b['flow_fw_valid'], b['flow_bw_valid']])
        terms['flow'], stats['flow_epe'] = sequence_loss(out['flow_preds'], gt_flow, gt_fv, L['gamma'])
        gt_disp = torch.cat([b['disp_t'], b['disp_t1']])
        gt_dv = torch.cat([b['disp_t_valid'], b['disp_t1_valid']])
        terms['disp'], stats['disp_epe'] = sequence_loss(out['disp_preds'], gt_disp, gt_dv, L['gamma'], max_mag=300)
    if supervised:
        terms['pose_sup'], ps = pose_supervised_loss(out['pose'], out['pose_logvar'], b['pose'])
        stats.update(ps)
    if not supervised and not frozen:
        if L['flow_photo']:
            terms['flow_photo'], _, _ = flow_photometric_loss(b['Lt'], b['Lt1'], flow_fw, flow_bw)
        if L['stereo_photo']:
            terms['stereo_photo'] = (stereo_photometric_loss(b['Lt'], b['Rt'], out['disp_t'].float())
                                     + stereo_photometric_loss(b['Lt1'], b['Rt1'], out['disp_t1'].float()))
        if L['flow_smooth']:
            terms['flow_smooth'] = smoothness(flow_fw, b['Lt']) + smoothness(flow_bw, b['Lt1'])
        if L['disp_smooth']:
            terms['disp_smooth'] = smoothness(out['disp_t'].float(), b['Lt'], order=2)
    if not supervised and L['pose_rigid']:
        terms['pose_rigid'] = pose_rigid_flow_loss(out, occ4)

    if L['seg'] and it >= L['seg_start']:
        img4 = F.avg_pool2d(b['Lt'], 4)
        terms['seg'], ss, (R, t) = motion_seg_loss(out, img4, occ4)
        stats.update(ss)
        if L['pose_kabsch'] and not supervised:   # pose head vs robust geometric ego-motion
            terms['pose_kabsch'] = pose_kabsch_consistency(out, R, t)

    total = sum(L[k] * v for k, v in terms.items())
    stats.update({f'loss_{k}': v.detach() for k, v in terms.items()})
    return total, stats


@torch.no_grad()
def validate(model, loader, max_batches=50):
    model.eval()
    acc = {}
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        b = to_cuda(b)
        out = forward(model, b)
        res = {}
        res.update({f'flow_{k}': v for k, v in flow_metrics(out['flow_fw'].float(), b['flow_fw'], b['flow_fw_valid']).items()})
        res.update({f'disp_{k}': v for k, v in disp_metrics(out['disp_t'].float(), b['disp_t'], b['disp_t_valid']).items()})
        res.update({f'pose_{k}': v for k, v in pose_metrics(out['pose'].float(), b['pose']).items()})
        if 'moving_t' in b and b['moving_t'].any():
            res.update(moving_iou(out['masks'].float(), b['moving_t']))
        for k, v in res.items():
            if not math.isnan(v):
                acc.setdefault(k, []).append(v)
    model.train()
    return {k: float(np.mean(v)) for k, v in acc.items()}


def log_images(writer, out, b, it, tag):
    i = 0
    img = to_np(b['Lt'][i])
    writer.add_image(f'{tag}/image', img, it)
    writer.add_image(f'{tag}/flow', flow_to_color(to_np(out['flow_fw'][i])), it, dataformats='HWC')
    writer.add_image(f'{tag}/disp', disp_to_color(to_np(out['disp_t'][i])), it, dataformats='HWC')
    writer.add_image(f'{tag}/masks', masks_to_color(to_np(out['masks'][i]), img), it, dataformats='HWC')


def param_groups(model, cfg):
    branch = [p for n, p in model.named_parameters() if n.startswith(('flow.', 'stereo.')) and p.requires_grad]
    heads = [p for n, p in model.named_parameters() if not n.startswith(('flow.', 'stereo.')) and p.requires_grad]
    groups = [dict(params=heads, name='heads')]
    if branch:
        groups.append(dict(params=branch, name='branch'))
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--set', nargs='*', default=[])
    ap.add_argument('--resume', default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    T, D, M = cfg['train'], cfg['data'], cfg['model']
    torch.manual_seed(cfg['seed'])
    run_dir = os.path.join('runs', cfg['name'])
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    writer = SummaryWriter(run_dir)

    model = StereoFlowFormer(num_slots=M['num_slots'], share_backbone=M['share_backbone']).cuda()
    frozen = bool(T.get('freeze_branches', False))
    if M.get('init'):
        sd = torch.load(M['init'], map_location='cpu', weights_only=False)['model']
        print('init from', M['init'], model.load_state_dict(sd, strict=False))
    else:
        model.load_pretrained(M['flow_ckpt'], M['stereo_ckpt'])
    model.set_freeze_branches(frozen)
    model.train()

    opt = torch.optim.AdamW(param_groups(model, cfg), lr=T['lr'], weight_decay=T['weight_decay'])

    def lr_lambda(group):
        def f(it):
            warm = min(1.0, (it + 1) / T['warmup'])
            cos = 0.5 * (1 + math.cos(math.pi * min(it, T['iters']) / T['iters']))
            mult = 1.0
            if group == 'branch' and it < T['branch_warm_iters']:
                mult = T['branch_lr_mult']
            return warm * (0.05 + 0.95 * cos) * mult
        return f
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lr_lambda(g['name']) for g in opt.param_groups])

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start = ck['it'] + 1

    crop = tuple(D['crop'])
    scenes = tuple(D.get('vkitti2_scenes') or VKITTI2_SCENES)
    vk_train = VKITTI2(D['vkitti2_root'], 'train', crop, scenes=scenes, dense_gt=not frozen, seed=cfg['seed'])
    vk_val = VKITTI2(D['vkitti2_root'], 'val', crop, scenes=scenes, augment=False, with_instances=True)
    vk_val = Subset(vk_val, list(range(0, len(vk_val), max(1, len(vk_val) // 100))))
    selfsup = 'kitti_root' in D
    if selfsup:
        kt = KITTIOdometry(D['kitti_root'], D['kitti_seqs'], crop, seed=cfg['seed'])
        kt_iter = infinite(make_loader(kt, T['batch_size'], D['num_workers']))
        vk_bs = T['batch_size']
    vk_iter = infinite(make_loader(vk_train, T['batch_size'], D['num_workers']))
    val_loader = make_loader(vk_val, 2, 0, shuffle=False)
    rng = np.random.default_rng(cfg['seed'])
    print(f'train samples: vkitti2 {len(vk_train)}' + (f', kitti {len(kt)}' if selfsup else ''))

    t0 = time.time()
    for it in range(start, T['iters']):
        supervised = (not selfsup) or rng.random() < D.get('vkitti_mix', 0.0)
        b = to_cuda(next(vk_iter) if supervised else next(kt_iter))
        out = forward(model, b)
        loss, stats = compute_losses(out, b, cfg, it, supervised, frozen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], T['grad_clip'])
        opt.step()
        sched.step()

        if it % T['log_every'] == 0:
            dt = (time.time() - t0) / (1 if it == start else T['log_every'])
            t0 = time.time()
            tag = 'sup' if supervised else 'self'
            writer.add_scalar(f'{tag}/loss', loss.item(), it)
            writer.add_scalar('train/grad_norm', gnorm.item(), it)
            writer.add_scalar('train/sec_per_it', dt, it)
            for g in opt.param_groups:
                writer.add_scalar(f'lr/{g["name"]}', g['lr'], it)
            for k, v in stats.items():
                writer.add_scalar(f'{tag}/{k}', float(v), it)
            msg = ' '.join(f'{k}={float(v):.3f}' for k, v in stats.items())
            print(f'[{it:6d}] {tag} loss={loss.item():.3f} {msg} ({dt:.2f}s/it)', flush=True)
        if it % T['vis_every'] == 0:
            log_images(writer, out, b, it, 'sup' if supervised else 'self')
        if (it + 1) % T['val_every'] == 0 or it == T['iters'] - 1:
            res = validate(model, val_loader)
            for k, v in res.items():
                writer.add_scalar(f'val/{k}', v, it)
            print('[val]', ' '.join(f'{k}={v:.3f}' for k, v in res.items()), flush=True)
        if (it + 1) % T['ckpt_every'] == 0 or it == T['iters'] - 1:
            ck = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), it=it, cfg=cfg)
            torch.save(ck, os.path.join(run_dir, 'checkpoints', 'latest.pth'))
            if (it + 1) % (T['ckpt_every'] * 5) == 0:
                torch.save(dict(model=model.state_dict(), it=it, cfg=cfg),
                           os.path.join(run_dir, 'checkpoints', f'step_{it + 1:06d}.pth'))
    writer.close()


if __name__ == '__main__':
    main()
