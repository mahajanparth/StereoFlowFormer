"""Config loading and visualisation helpers."""
import cv2
import numpy as np
import torch
import yaml


def load_config(path, overrides=()):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if 'base' in cfg:
        base = load_config(cfg.pop('base'))
        cfg = _merge(base, cfg)
    for ov in overrides:
        key, val = ov.split('=', 1)
        node = cfg
        *parents, leaf = key.split('.')
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(val)
    return cfg


def _merge(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _color_wheel():
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    cols = []
    for n, (a, b) in ((RY, (0, 1)), (YG, (1, 0)), (GC, (1, 2)), (CB, (2, 1)), (BM, (2, 0)), (MR, (0, 2))):
        for i in range(n):
            c = [0, 0, 0]
            if (a, b) in ((0, 1), (1, 2), (2, 0)):
                c[a], c[b] = 255, int(255 * i / n)
            else:
                c[b], c[a] = 255, 255 - int(255 * i / n)
            cols.append(c)
    return np.array(cols, np.float32)


_WHEEL = _color_wheel()


def flow_to_color(flow, max_mag=None):
    """flow [2,H,W] numpy -> uint8 RGB [H,W,3] (Middlebury colour coding)."""
    u, v = flow[0], flow[1]
    mag = np.sqrt(u ** 2 + v ** 2)
    max_mag = max_mag or max(np.percentile(mag, 99), 1e-3)
    u, v = u / max_mag, v / max_mag
    rad = np.clip(np.sqrt(u ** 2 + v ** 2), 0, 1)
    ang = np.arctan2(-v, -u) / np.pi
    fk = (ang + 1) / 2 * (len(_WHEEL) - 1)
    k0 = np.floor(fk).astype(int)
    k1 = (k0 + 1) % len(_WHEEL)
    f = (fk - k0)[..., None]
    col = (1 - f) * _WHEEL[k0] + f * _WHEEL[k1]
    col = 255 - rad[..., None] * (255 - col)
    return col.astype(np.uint8)


def disp_to_color(disp, vmax=None):
    d = disp.squeeze()
    vmax = vmax or max(np.percentile(d, 99), 1e-3)
    return cv2.applyColorMap((np.clip(d / vmax, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)[..., ::-1]


PALETTE = np.array([[40, 40, 40], [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
                    [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60]], np.uint8)


def masks_to_color(masks, img=None, alpha=0.5):
    """masks [K,H,W] soft -> RGB overlay of the argmax slot."""
    lab = masks.argmax(0)
    col = PALETTE[lab % len(PALETTE)]
    if img is not None:
        base = (img.transpose(1, 2, 0) * 255).astype(np.float32)
        col = (alpha * col + (1 - alpha) * base).astype(np.uint8)
    return col


def to_np(x):
    return x.detach().float().cpu().numpy() if torch.is_tensor(x) else x
