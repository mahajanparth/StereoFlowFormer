"""Stereo VO + bundle adjustment + map on a KITTI odometry sequence.

  python scripts/run_vo.py --seq 09 --ckpt runs/stage2_kitti_selfsup/checkpoints/latest.pth
  python scripts/run_vo.py --seq 09 --ckpt ... --modes pose_head pnp pnp_ba pnp_ba_prior   # ablation

Network outputs are computed once per frame pair and reused for every back-end mode.
Outputs (results/<seq>/): <mode>_traj.txt (KITTI format), metrics.json, map_<mode>.ply, plot.png
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvo.data.kitti import KITTIOdometry, read_kitti_poses  # noqa: E402
from dvo.eval.traj_eval import evaluate, save_kitti  # noqa: E402
from dvo.slam.frontend import infer_pair, load_model  # noqa: E402
from dvo.slam.mapping import PointMap, keyframe_points  # noqa: E402
from dvo.slam.vo import StereoVO, VOConfig  # noqa: E402


def gt_poses_cam2(gt_path, K, P2_tx):
    """KITTI GT is for cam0; shift to the colour-left camera (cam2): T_w_c2 = T_w_c0 @ T_c0_c2."""
    gt = read_kitti_poses(gt_path)
    T = np.eye(4)
    T[0, 3] = P2_tx
    return np.linalg.inv(gt[0] @ T) @ gt @ T  # world = first colour-left camera, as in the VO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='09')
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--kitti_root', default='data/kitti_odometry')
    ap.add_argument('--poses_root', default='data/kitti_odometry_poses/dataset/poses')
    ap.add_argument('--modes', nargs='*', default=['pnp_ba_prior'])
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--no_masks', action='store_true')
    ap.add_argument('--out', default='results')
    ap.add_argument('--map', action='store_true', help='fuse a coloured point-cloud map for the last mode')
    args = ap.parse_args()

    out_dir = os.path.join(args.out, args.seq)
    os.makedirs(out_dir, exist_ok=True)
    ds = KITTIOdometry(args.kitti_root, [args.seq], crop_hw=None, augment=False)
    meta = ds.seq_meta[args.seq]
    n = meta['n'] if args.max_frames is None else min(meta['n'], args.max_frames)
    K, bl = meta['K'], float(meta['baseline'])

    model = load_model(args.ckpt, pretrained=None if args.ckpt else (
        'checkpoints/unimatch/gmflow-scale2-regrefine6-things-776ed612.pth',
        'checkpoints/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth'))

    # ---- network pass (kept on CPU in fp16 to bound memory)
    t0 = time.time()
    nets = []
    for t in range(n - 1):
        s = ds.load(args.seq, t, t + 1)
        net = infer_pair(model, s)
        nets.append({k: (v.half().cpu() if v.dim() > 1 else v.cpu()) for k, v in net.items()})
        if t % 200 == 0:
            print(f'net {t}/{n - 1} ({(time.time() - t0) / (t + 1):.3f}s/pair)', flush=True)
    H, W = nets[0]['disp_t'].shape[-2:]

    gt = None
    gt_file = os.path.join(args.poses_root, f'{args.seq}.txt')
    if os.path.exists(gt_file):
        P2_tx = 0.0  # odometry calib P2[0,3]/fx is the cam0 -> cam2 x-offset
        with open(os.path.join(meta['dir'], 'calib.txt')) as f:
            for line in f:
                if line.startswith('P2:'):
                    P = np.array(line.split()[1:], float).reshape(3, 4)
                    P2_tx = P[0, 3] / P[0, 0]
        gt = gt_poses_cam2(gt_file, K, P2_tx)[:n]

    results = {}
    for mode in args.modes:
        cfg = VOConfig(mode=mode, use_masks=not args.no_masks)
        vo = StereoVO(K, bl, (H, W), cfg)
        t1 = time.time()
        for t, net in enumerate(nets):
            net = {k: v.float().cuda() for k, v in net.items()}
            if t == 0:
                vo.start(net)
            vo.step(net)
        vo.finish()
        traj = vo.trajectory_wc()
        save_kitti(os.path.join(out_dir, f'{mode}_traj.txt'), traj)
        res = dict(keyframes=len(vo.kf_frames), landmarks=len(vo.lm_X), backend_sec=time.time() - t1)
        if gt is not None:
            res.update(evaluate(gt, traj))
        results[mode] = res
        print(mode, json.dumps(res), flush=True)

        if args.map and mode == args.modes[-1]:
            pmap = PointMap(voxel=0.1)
            for kf_frame in vo.kf_frames:
                s = ds.load(args.seq, kf_frame, min(kf_frame + 1, n - 1))
                net = nets[min(kf_frame, len(nets) - 1)]
                static = net['masks'][0].float().numpy() if not args.no_masks else None
                pts, cols = keyframe_points(s['Lt'], net['disp_t'][0].float().numpy(), K, bl, static)
                pmap.add(pts, cols, np.linalg.inv(vo.poses[kf_frame]))
            npts = pmap.save(os.path.join(out_dir, f'map_{mode}.ply'))
            print(f'map: {npts} points -> {out_dir}/map_{mode}.ply')

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(results, f, indent=2)
    if gt is not None:
        plot(out_dir, gt, {m: np.loadtxt(os.path.join(out_dir, f'{m}_traj.txt')) for m in args.modes})


def plot(out_dir, gt, trajs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 7))
    plt.plot(gt[:, 0, 3], gt[:, 2, 3], 'k-', label='GT', lw=2)
    for m, tr in trajs.items():
        tr = tr.reshape(-1, 3, 4)
        plt.plot(tr[:, 0, 3], tr[:, 2, 3], label=m)
    plt.axis('equal')
    plt.legend()
    plt.xlabel('x [m]')
    plt.ylabel('z [m]')
    plt.savefig(os.path.join(out_dir, 'plot.png'), dpi=120, bbox_inches='tight')


if __name__ == '__main__':
    main()
