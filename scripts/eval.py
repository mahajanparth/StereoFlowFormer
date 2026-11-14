"""Evaluate flow / disparity / motion segmentation / pose.

  python scripts/eval.py --dataset kitti2015 [--ckpt runs/.../latest.pth]
  python scripts/eval.py --dataset vkitti2   [--ckpt ...]
Without --ckpt the UniMatch-initialised model is evaluated (heads untrained).
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvo.data.kitti import KITTI2015  # noqa: E402
from dvo.data.vkitti2 import VKITTI2  # noqa: E402
from dvo.eval.metrics import disp_metrics, fg_ari, flow_metrics, moving_iou, pose_metrics  # noqa: E402
from dvo.slam.frontend import load_model  # noqa: E402

PRETRAINED = ('checkpoints/unimatch/gmflow-scale2-regrefine6-things-776ed612.pth',
              'checkpoints/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth')


@torch.no_grad()
def run(model, b):
    imgs = [b[k][None].cuda() for k in ('Lt', 'Rt', 'Lt1', 'Rt1')]
    H, W = imgs[0].shape[-2:]
    ph, pw = (-H) % 32, (-W) % 32
    imgs = [F.pad(x, (0, pw, 0, ph), mode='replicate') for x in imgs]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(*imgs, b['K'][None].cuda(), b['baseline'].view(1).cuda())
    crop = {k: out[k][..., :H, :W].float() for k in ('flow_fw', 'disp_t', 'masks')}
    crop['pose'] = out['pose'].float()
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['kitti2015', 'vkitti2'], default='kitti2015')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--root', default=None)
    ap.add_argument('--max', type=int, default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    model = load_model(args.ckpt, pretrained=None if args.ckpt else PRETRAINED)
    if args.dataset == 'kitti2015':
        ds = KITTI2015(args.root or 'data/kitti2015')
    else:
        ds = VKITTI2(args.root or 'data/vkitti2', 'val', crop_hw=None, augment=False, with_instances=True)
    n = len(ds) if args.max is None else min(args.max, len(ds))
    acc = {}
    for i in range(n):
        b = ds[i]
        out = run(model, b)
        g = {k: v[None].cuda() for k, v in b.items() if torch.is_tensor(v) and v.dim() >= 2}
        res = {f'flow_{k}': v for k, v in flow_metrics(out['flow_fw'], g['flow_fw'], g['flow_fw_valid']).items()}
        res.update({f'disp_{k}': v for k, v in disp_metrics(out['disp_t'], g['disp_t'], g['disp_t_valid']).items()})
        if g['moving_t'].any():
            res.update(moving_iou(out['masks'], g['moving_t']))
            if 'inst_t' in g:
                try:
                    res.update(fg_ari(out['masks'], g['inst_t'], g['moving_t']))
                except ImportError:
                    pass
        if 'pose' in b:
            res.update({f'pose_{k}': v for k, v in pose_metrics(out['pose'], b['pose'][None].cuda()).items()})
        for k, v in res.items():
            if not math.isnan(v):
                acc.setdefault(k, []).append(v)
    summary = {k: float(np.mean(v)) for k, v in acc.items()}
    print(json.dumps(summary, indent=2))
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(summary, f, indent=2)


if __name__ == '__main__':
    main()
