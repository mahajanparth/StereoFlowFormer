"""Pinhole stereo camera helpers. Intrinsics are [B, 3, 3] tensors in pixels."""
import torch


def disp_to_depth(disp, fx, baseline, min_disp=1e-3):
    """depth = fx * B / disp. disp [B,1,H,W] in pixels, fx/baseline [B] or float."""
    fx = _bcast(fx, disp)
    baseline = _bcast(baseline, disp)
    return fx * baseline / disp.clamp(min=min_disp)


def depth_to_disp(depth, fx, baseline, min_depth=1e-3):
    fx = _bcast(fx, depth)
    baseline = _bcast(baseline, depth)
    return fx * baseline / depth.clamp(min=min_depth)


def scale_intrinsics(K, sx, sy=None):
    """Scale intrinsics for an image resized by (sx, sy)."""
    sy = sx if sy is None else sy
    K = K.clone()
    K[..., 0, :] = K[..., 0, :] * sx
    K[..., 1, :] = K[..., 1, :] * sy
    return K


def crop_intrinsics(K, x0, y0):
    K = K.clone()
    K[..., 0, 2] = K[..., 0, 2] - x0
    K[..., 1, 2] = K[..., 1, 2] - y0
    return K


def _bcast(v, ref):
    if not torch.is_tensor(v):
        return v
    return v.view(-1, *([1] * (ref.dim() - 1))).to(ref)
