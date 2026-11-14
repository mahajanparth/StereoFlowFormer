"""KITTI odometry (stereo sequences, self-supervised training + VO eval) and KITTI 2015 (flow/disp/obj eval)."""
import glob
import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from .common import color_jitter, crop, random_crop, read_rgb, to_torch


def read_odometry_calib(path):
    P = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                P[k.strip()] = np.array(v.split(), np.float64).reshape(3, 4)
    P2, P3 = P['P2'], P['P3']
    K = P2[:, :3].astype(np.float32)
    baseline = (P2[0, 3] - P3[0, 3]) / P2[0, 0]
    return K, np.float32(baseline)


def read_kitti_poses(path):
    """Ground-truth poses: each line is T_w_cam0 (3x4). Returns [N,4,4]."""
    rows = np.loadtxt(path).reshape(-1, 3, 4)
    T = np.tile(np.eye(4), (len(rows), 1, 1))
    T[:, :3] = rows
    return T


class KITTIOdometry(Dataset):
    """Consecutive stereo pairs from KITTI odometry colour sequences (image_2 = left, image_3 = right)."""

    def __init__(self, root, seqs, crop_hw=(320, 960), augment=True, stride=1, seed=0):
        self.root, self.crop_hw, self.augment = root, crop_hw, augment
        self.rng = np.random.default_rng(seed)
        self.items, self.seq_meta = [], {}
        for seq in seqs:
            sdir = os.path.join(root, 'sequences', seq)
            left = sorted(glob.glob(os.path.join(sdir, 'image_2', '*.*')))
            if not left:
                continue
            K, bl = read_odometry_calib(os.path.join(sdir, 'calib.txt'))
            self.seq_meta[seq] = dict(dir=sdir, K=K, baseline=bl, ext=os.path.splitext(left[0])[1], n=len(left))
            self.items += [(seq, t) for t in range(0, len(left) - stride, 1)]
        self.stride = stride
        if not self.items:
            raise RuntimeError(f'No KITTI odometry data under {root} for {seqs}')

    def __len__(self):
        return len(self.items)

    def frame_paths(self, seq, t):
        m = self.seq_meta[seq]
        return (os.path.join(m['dir'], 'image_2', f'{t:06d}{m["ext"]}'),
                os.path.join(m['dir'], 'image_3', f'{t:06d}{m["ext"]}'))

    def load(self, seq, t, t1=None):
        t1 = t + self.stride if t1 is None else t1
        m = self.seq_meta[seq]
        (l0, r0), (l1, r1) = self.frame_paths(seq, t), self.frame_paths(seq, t1)
        return dict(Lt=read_rgb(l0), Rt=read_rgb(r0), Lt1=read_rgb(l1), Rt1=read_rgb(r1),
                    K=m['K'].copy(), baseline=m['baseline'])

    def __getitem__(self, idx):
        seq, t = self.items[idx]
        s = self.load(seq, t)
        if self.crop_hw is not None:
            s = random_crop(s, self.crop_hw, self.rng)
        if self.augment:
            s = color_jitter(s, self.rng)
        return to_torch(s)

    def worker_init(self, worker_id):
        self.rng = np.random.default_rng(np.random.SeedSequence([worker_id, os.getpid()]))


def read_kitti_flow(path):
    raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32)
    flow = (raw[..., :2] - 2 ** 15) / 64.0
    return flow.transpose(2, 0, 1), raw[..., 2:3].transpose(2, 0, 1)


def read_kitti_disp(path):
    d = cv2.imread(path, cv2.IMREAD_ANYDEPTH).astype(np.float32) / 256.0
    return d[None], (d > 0)[None].astype(np.float32)


def read_cam_to_cam(path):
    vals = {}
    with open(path) as f:
        for line in f:
            if ':' in line:
                k, v = line.split(':', 1)
                try:
                    vals[k] = np.array(v.split(), np.float64)
                except ValueError:
                    pass
    P2, P3 = vals['P_rect_02'].reshape(3, 4), vals['P_rect_03'].reshape(3, 4)
    return P2[:, :3].astype(np.float32), np.float32((P2[0, 3] - P3[0, 3]) / P2[0, 0])


class KITTI2015(Dataset):
    """KITTI 2015 scene-flow training split (200 samples) for evaluation."""

    def __init__(self, root):
        self.root = os.path.join(root, 'training')
        self.ids = sorted(os.path.basename(p)[:6] for p in glob.glob(os.path.join(self.root, 'image_2', '*_10.png')))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid, r = self.ids[i], self.root
        K, bl = read_cam_to_cam(os.path.join(r, '..', 'training', 'calib_cam_to_cam', f'{sid}.txt'))
        s = dict(Lt=read_rgb(f'{r}/image_2/{sid}_10.png'), Rt=read_rgb(f'{r}/image_3/{sid}_10.png'),
                 Lt1=read_rgb(f'{r}/image_2/{sid}_11.png'), Rt1=read_rgb(f'{r}/image_3/{sid}_11.png'),
                 K=K, baseline=bl)
        s['flow_fw'], s['flow_fw_valid'] = read_kitti_flow(f'{r}/flow_occ/{sid}_10.png')
        s['disp_t'], s['disp_t_valid'] = read_kitti_disp(f'{r}/disp_occ_0/{sid}_10.png')
        obj = cv2.imread(f'{r}/obj_map/{sid}_10.png', cv2.IMREAD_ANYDEPTH)
        s['moving_t'] = (obj > 0)[None]
        s['id'] = sid
        return to_torch(s)
