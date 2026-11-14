"""Virtual KITTI 2 (clone variant): stereo RGB, GT flow (fw/bw), depth -> disparity, instances, poses.

Layout (after scripts/download_data.py):
  data/vkitti2/SceneXX/clone/{intrinsic,extrinsic,bbox,info}.txt
  data/vkitti2/SceneXX/clone/frames/rgb/Camera_{0,1}/rgb_%05d.jpg
  data/vkitti2/SceneXX/clone/frames/depth/Camera_0/depth_%05d.png               (uint16, cm)
  data/vkitti2/SceneXX/clone/frames/forwardFlow/Camera_0/flow_%05d.png           (t -> t+1)
  data/vkitti2/SceneXX/clone/frames/backwardFlow/Camera_0/backwardFlow_%05d.png  (t -> t-1)
  data/vkitti2/SceneXX/clone/frames/instanceSegmentation/Camera_0/instancegt_%05d.png
"""
import os

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .common import color_jitter, random_crop, read_rgb, to_torch

SCENES = ('Scene01', 'Scene02', 'Scene06', 'Scene18', 'Scene20')
BASELINE = 0.532725
MAX_DEPTH = 655.0  # sky is encoded as 655.35 m


def read_vkitti_flow(path):
    bgr = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if bgr is None:
        raise FileNotFoundError(path)
    h, w, _ = bgr.shape
    valid = bgr[..., 0] != 0
    flow = 2.0 / (2 ** 16 - 1.0) * bgr[..., 2:0:-1].astype(np.float32) - 1
    flow[..., 0] *= w - 1
    flow[..., 1] *= h - 1
    flow[~valid] = 0
    return flow.transpose(2, 0, 1), valid[None].astype(np.float32)


def read_vkitti_depth(path):
    d = cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise FileNotFoundError(path)
    return d.astype(np.float32)[None] / 100.0


def read_vkitti_instances(path):
    """Palette PNG of instance ids (0 = background, else trackID + 1); None if missing."""
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path)).astype(np.int64)


def read_table(path):
    with open(path) as f:
        header = f.readline().split()
        rows = [[1.0 if v == 'True' else 0.0 if v == 'False' else float(v) for v in line.split()]
                for line in f if line.strip()]
    return header, np.array(rows)


class VKITTI2(Dataset):
    def __init__(self, root, split='train', crop_hw=(320, 960), scenes=SCENES, variant='clone', val_frames=50,
                 augment=True, with_instances=False, dense_gt=True, seed=0):
        self.root, self.crop_hw, self.augment = root, crop_hw, augment and split == 'train'
        self.dense_gt = dense_gt  # flow / disparity GT (not needed when the matching branches are frozen)
        self.with_instances = with_instances
        self.rng = np.random.default_rng(seed)
        self.items, self.meta = [], {}
        for scene in scenes:
            sdir = os.path.join(root, scene, variant)
            if not os.path.isdir(sdir):
                continue
            K0, E = self._load_cams(sdir)
            moving = self._load_moving(sdir)
            n = len(E[0])
            self.meta[scene] = dict(dir=sdir, K=K0, E=E, moving=moving)
            frames = range(n - 1 - val_frames) if split == 'train' else range(n - 1 - val_frames, n - 1)
            self.items += [(scene, t) for t in frames]
        if not self.items:
            raise RuntimeError(f'No VKITTI2 data found under {root}')

    @staticmethod
    def _load_cams(sdir):
        _, intr = read_table(os.path.join(sdir, 'intrinsic.txt'))   # frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        _, extr = read_table(os.path.join(sdir, 'extrinsic.txt'))   # frame cameraID r1,1 ... t3 0 0 0 1
        c0 = intr[intr[:, 1] == 0][0]
        K = np.array([[c0[2], 0, c0[4]], [0, c0[3], c0[5]], [0, 0, 1]], np.float32)
        E = {}
        for cam in (0, 1):
            rows = extr[extr[:, 1] == cam]
            rows = rows[np.argsort(rows[:, 0])]
            E[cam] = rows[:, 2:18].reshape(-1, 4, 4)                 # world -> camera
        return K, E

    @staticmethod
    def _load_moving(sdir):
        path = os.path.join(sdir, 'bbox.txt')
        if not os.path.exists(path):
            return {}
        header, rows = read_table(path)
        fi, ci, ti, mi = (header.index(k) for k in ('frame', 'cameraID', 'trackID', 'isMoving'))
        moving = {}
        for r in rows[rows[:, ci] == 0]:
            if r[mi] > 0:
                moving.setdefault(int(r[fi]), set()).add(int(r[ti]))
        return moving

    def __len__(self):
        return len(self.items)

    def _path(self, sdir, kind, cam, name, t):
        return os.path.join(sdir, 'frames', kind, f'Camera_{cam}', f'{name}_{t:05d}.{"jpg" if kind == "rgb" else "png"}')

    def load(self, idx):
        scene, t = self.items[idx]
        m = self.meta[scene]
        d = m['dir']
        s = dict(
            Lt=read_rgb(self._path(d, 'rgb', 0, 'rgb', t)), Rt=read_rgb(self._path(d, 'rgb', 1, 'rgb', t)),
            Lt1=read_rgb(self._path(d, 'rgb', 0, 'rgb', t + 1)), Rt1=read_rgb(self._path(d, 'rgb', 1, 'rgb', t + 1)),
            K=m['K'].copy(), baseline=np.float32(BASELINE),
        )
        fx = m['K'][0, 0]
        if self.dense_gt:
            s['flow_fw'], s['flow_fw_valid'] = read_vkitti_flow(self._path(d, 'forwardFlow', 0, 'flow', t))
            s['flow_bw'], s['flow_bw_valid'] = read_vkitti_flow(self._path(d, 'backwardFlow', 0, 'backwardFlow', t + 1))
            for key, tt in (('disp_t', t), ('disp_t1', t + 1)):
                depth = read_vkitti_depth(self._path(d, 'depth', 0, 'depth', tt))
                s[key] = (fx * BASELINE / depth).astype(np.float32)
                s[key + '_valid'] = (depth < MAX_DEPTH).astype(np.float32)
        E0 = m['E'][0]
        s['pose'] = (E0[t + 1] @ np.linalg.inv(E0[t])).astype(np.float32)
        inst = read_vkitti_instances(self._path(d, 'instanceSegmentation', 0, 'instancegt', t)) \
            if self.with_instances else None
        if inst is not None:                                    # 0 = bg, else trackID + 1
            mv = np.zeros_like(inst, dtype=bool)
            for tid in m['moving'].get(t, ()):
                mv |= inst == tid + 1
            s['inst_t'], s['moving_t'] = inst[None], mv[None]
        return s

    def __getitem__(self, idx):
        s = self.load(idx)
        if self.crop_hw is not None:
            s = random_crop(s, self.crop_hw, self.rng) if self.augment else _center_crop(s, self.crop_hw)
        if self.augment:
            s = color_jitter(s, self.rng)
        return to_torch(s)

    def worker_init(self, worker_id):
        self.rng = np.random.default_rng(np.random.SeedSequence([worker_id, os.getpid()]))


def _center_crop(s, crop_hw):
    from .common import crop
    H, W = s['Lt'].shape[-2:]
    ch, cw = crop_hw
    return crop(s, H - ch, (W - cw) // 2, ch, cw)  # keep the bottom (road) part
