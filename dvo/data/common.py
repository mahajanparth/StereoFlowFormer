"""Shared sample format and augmentation for 4-view (two stereo pairs) samples.

A sample is a dict of numpy/torch arrays:
  Lt, Rt, Lt1, Rt1   float32 [3,H,W] in [0,1]
  K                  float32 [3,3] left rectified intrinsics
  baseline           float32 scalar (metres)
optional ground truth (any subset):
  flow_fw, flow_bw   [2,H,W] (L_t->L_t+1 and L_t+1->L_t), flow_fw_valid / flow_bw_valid [1,H,W]
  disp_t, disp_t1    [1,H,W] left disparity (px), disp_t_valid / disp_t1_valid
  pose               [4,4] T_t->t+1 (camera-t points into camera-(t+1))
  inst_t             [1,H,W] int instance ids (0 = background), moving_t [1,H,W] bool
"""
import cv2
import numpy as np
import torch

IMAGE_KEYS = ('Lt', 'Rt', 'Lt1', 'Rt1')
DENSE_KEYS = ('flow_fw', 'flow_bw', 'flow_fw_valid', 'flow_bw_valid', 'disp_t', 'disp_t1', 'disp_t_valid',
              'disp_t1_valid', 'inst_t', 'moving_t')


def read_rgb(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).astype(np.float32) / 255.0


def random_crop(sample, crop_hw, rng):
    H, W = sample['Lt'].shape[-2:]
    ch, cw = crop_hw
    assert H >= ch and W >= cw, (H, W, crop_hw)
    y0 = rng.integers(0, H - ch + 1)
    x0 = rng.integers(0, W - cw + 1)
    return crop(sample, y0, x0, ch, cw)


def crop(sample, y0, x0, ch, cw):
    out = dict(sample)
    for k in IMAGE_KEYS + DENSE_KEYS:
        if k in out:
            out[k] = out[k][..., y0:y0 + ch, x0:x0 + cw]
    K = out['K'].copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    out['K'] = K
    return out


def color_jitter(sample, rng, strength=0.2):
    """Same photometric jitter on all four views (keeps brightness constancy for self-supervision)."""
    b = 1 + rng.uniform(-strength, strength)
    c = 1 + rng.uniform(-strength, strength)
    s = 1 + rng.uniform(-strength, strength)
    gamma = 1 + rng.uniform(-0.1, 0.1)
    out = dict(sample)
    for k in IMAGE_KEYS:
        img = out[k] * b
        mean = img.mean()
        img = (img - mean) * c + mean
        gray = img.mean(0, keepdims=True)
        img = (img - gray) * s + gray
        out[k] = np.clip(img, 0, 1) ** gamma
    return out


def to_torch(sample):
    res = {}
    for k, v in sample.items():
        if isinstance(v, np.ndarray):
            res[k] = torch.from_numpy(np.ascontiguousarray(v))
        elif isinstance(v, (float, int, np.floating)):
            res[k] = torch.tensor(v, dtype=torch.float32)
        else:
            res[k] = v
    return res
