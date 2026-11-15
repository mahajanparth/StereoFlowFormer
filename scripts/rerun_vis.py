"""Stream network outputs and online stereo VO on a KITTI odometry sequence to the Rerun viewer.

  python scripts/rerun_vis.py --seq 09 --max_frames 300
  python scripts/rerun_vis.py --seq 09 --save vis/seq09.rrd      # write a recording, open with `rerun vis/seq09.rrd`

Logged per frame: left image with motion masks, flow, disparity, depth (unprojected in 3D), the camera,
the estimated and GT trajectories, and a coloured static-background point cloud for each keyframe.
"""
import argparse
import os
import sys

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvo.data.kitti import KITTIOdometry  # noqa: E402
from dvo.slam.frontend import infer_pair, load_model  # noqa: E402
from dvo.slam.mapping import keyframe_points  # noqa: E402
from dvo.slam.vo import StereoVO, VOConfig  # noqa: E402
from dvo.utils import PALETTE, disp_to_color, flow_to_color, to_np  # noqa: E402
from run_vo import gt_poses_cam2  # noqa: E402

CAM = 'world/camera/image'


def blueprint():
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name='3D', origin='world', contents=['$origin/**', f'- {CAM}/masks']),
            rrb.Vertical(
                rrb.Spatial2DView(name='Left image', origin=CAM, contents=['$origin/rgb']),
                rrb.Spatial2DView(name='Optical flow (fw)', origin='views/flow'),
                rrb.Spatial2DView(name='Disparity', origin='views/disparity'),
                rrb.TimeSeriesView(name='Tracking', origin='stats'),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )


def p2_offset(seq_dir):
    with open(os.path.join(seq_dir, 'calib.txt')) as f:
        for line in f:
            if line.startswith('P2:'):
                P = np.array(line.split()[1:], float).reshape(3, 4)
                return P[0, 3] / P[0, 0]
    return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seq', default='09')
    ap.add_argument('--ckpt', default='runs/stage2b_kitti_heads/checkpoints/latest.pth')
    ap.add_argument('--kitti_root', default='data/kitti_odometry')
    ap.add_argument('--poses_root', default='data/kitti_odometry_poses/dataset/poses')
    ap.add_argument('--mode', default='pnp_ba_prior')
    ap.add_argument('--max_frames', type=int, default=300)
    ap.add_argument('--save', default=None, help='write an .rrd file instead of spawning the viewer')
    args = ap.parse_args()

    rr.init('stereoflowformer', spawn=args.save is None, default_blueprint=blueprint())
    if args.save:
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        rr.save(args.save, default_blueprint=blueprint())

    ds = KITTIOdometry(args.kitti_root, [args.seq], crop_hw=None, augment=False)
    meta = ds.seq_meta[args.seq]
    n = min(meta['n'], args.max_frames)
    K, bl = meta['K'], float(meta['baseline'])
    model = load_model(args.ckpt)

    rr.log('world', rr.ViewCoordinates.RDF, static=True)       # KITTI camera frame: x right, y down, z forward
    rr.log(CAM, rr.AnnotationContext([(0, 'background', (0, 0, 0, 0))] +
                                     [(i, f'object {i}', tuple(PALETTE[i])) for i in range(1, len(PALETTE))]),
           static=True)
    gt_file = os.path.join(args.poses_root, f'{args.seq}.txt')
    if os.path.exists(gt_file):
        gt = gt_poses_cam2(gt_file, K, p2_offset(meta['dir']))[:n]
        rr.log('world/gt_trajectory', rr.LineStrips3D([gt[:, :3, 3]], colors=[(160, 160, 160)], radii=0.15),
               static=True)

    vo = None
    n_kf = 0
    for t in range(n - 1):
        s = ds.load(args.seq, t, t + 1)
        net = infer_pair(model, s)
        if vo is None:
            H, W = net['disp_t'].shape[-2:]
            vo = StereoVO(K, bl, (H, W), VOConfig(mode=args.mode))
            vo.start(net)
        vo.step(net)

        rr.set_time('frame', sequence=t)
        rgb = (s['Lt'].transpose(1, 2, 0) * 255).astype(np.uint8)
        disp = to_np(net['disp_t'])[0]
        masks = to_np(net['masks'])
        depth = np.where(disp > 1.0, K[0, 0] * bl / np.maximum(disp, 1e-3), 0).astype(np.float32)
        depth[depth > 40] = 0
        depth[: depth.shape[0] // 4] = 0                         # sky / far field

        T_wc = np.linalg.inv(vo.poses[t])
        rr.log('world/camera', rr.Transform3D(translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3]))
        rr.log(CAM, rr.Pinhole(image_from_camera=K, resolution=[W, H], image_plane_distance=1.5))
        rr.log(f'{CAM}/rgb', rr.Image(rgb).compress(jpeg_quality=90))
        rr.log(f'{CAM}/masks', rr.SegmentationImage(masks.argmax(0).astype(np.uint8), opacity=0.5))
        rr.log(f'{CAM}/depth', rr.DepthImage((depth * 1000).astype(np.uint16), meter=1000.0, colormap='turbo',
                                               point_fill_ratio=0.3))
        rr.log('views/flow', rr.Image(flow_to_color(to_np(net['flow_fw']))).compress(jpeg_quality=90))
        rr.log('views/disparity', rr.Image(disp_to_color(disp)).compress(jpeg_quality=90))

        traj = vo.trajectory_wc()[:, :3, 3]
        rr.log('world/trajectory', rr.LineStrips3D([traj], colors=[(230, 25, 75)], radii=0.2))
        st = vo.stats[-1]
        rr.log('stats/tracks', rr.Scalars(st['tracks']))
        rr.log('stats/pnp_inliers', rr.Scalars(st['inliers']))

        # static-background cloud for each new keyframe (pose as estimated at the time)
        while n_kf < len(vo.kf_frames) and vo.kf_frames[n_kf] <= t:
            f = vo.kf_frames[n_kf]
            if f == t:
                pts, cols = keyframe_points(s['Lt'], disp, K, bl, masks[0], stride=4)
                pts = pts @ T_wc[:3, :3].T + T_wc[:3, 3]
                rr.log(f'world/map/kf_{n_kf:05d}', rr.Points3D(pts, colors=cols, radii=0.04))
            n_kf += 1
        if t % 50 == 0:
            print(f'frame {t}/{n - 1}', flush=True)

    vo.finish()
    rr.log('world/trajectory', rr.LineStrips3D([vo.trajectory_wc()[:, :3, 3]], colors=[(230, 25, 75)], radii=0.2))
    print('done' + (f', saved {args.save}' if args.save else ''))


if __name__ == '__main__':
    main()
