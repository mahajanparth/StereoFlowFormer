"""Render flow / disparity / motion-mask overlays for frames of a KITTI odometry sequence or KITTI 2015.

  python scripts/visualize.py --seq 09 --frames 0 100 200 [--ckpt ...] --out vis/
  python scripts/visualize.py --kitti2015 0 10 20 [--ckpt ...]
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dvo.data.kitti import KITTI2015, KITTIOdometry  # noqa: E402
from dvo.slam.frontend import infer_pair, load_model  # noqa: E402
from dvo.utils import disp_to_color, flow_to_color, masks_to_color, to_np  # noqa: E402

PRETRAINED = ('checkpoints/unimatch/gmflow-scale2-regrefine6-things-776ed612.pth',
              'checkpoints/unimatch/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth')


def panel(s, net):
    img = s['Lt']
    rows = [
        (img.transpose(1, 2, 0) * 255).astype(np.uint8),
        flow_to_color(to_np(net['flow_fw'])),
        disp_to_color(to_np(net['disp_t'])),
        masks_to_color(to_np(net['masks']), img),
    ]
    return np.concatenate(rows, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--seq', default=None)
    ap.add_argument('--frames', nargs='*', type=int, default=[0])
    ap.add_argument('--kitti2015', nargs='*', type=int, default=None)
    ap.add_argument('--kitti_root', default='data/kitti_odometry')
    ap.add_argument('--out', default='vis')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    model = load_model(args.ckpt, pretrained=None if args.ckpt else PRETRAINED)
    if args.kitti2015 is not None:
        ds = KITTI2015('data/kitti2015')
        for i in args.kitti2015:
            s = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in ds[i].items()}
            net = infer_pair(model, s)
            cv2.imwrite(os.path.join(args.out, f'kitti15_{i:03d}.png'), panel(s, net)[..., ::-1])
    if args.seq:
        ds = KITTIOdometry(args.kitti_root, [args.seq], crop_hw=None, augment=False)
        for t in args.frames:
            s = ds.load(args.seq, t, t + 1)
            net = infer_pair(model, s)
            cv2.imwrite(os.path.join(args.out, f'seq{args.seq}_{t:06d}.png'), panel(s, net)[..., ::-1])
    print('saved to', args.out)


if __name__ == '__main__':
    main()
