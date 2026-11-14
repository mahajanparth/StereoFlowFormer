"""Minimal SE(3) utilities (torch, batched).

Poses are 4x4 matrices T_ab mapping points from frame b to frame a: X_a = R X_b + t.
The relative motion T_t->t+1 used throughout maps camera-t points into camera-(t+1).
A 6-vector pose is (rotvec, t), decoupled, which is what the pose head regresses.
"""
import torch


def skew(w):
    wx, wy, wz = w.unbind(-1)
    o = torch.zeros_like(wx)
    return torch.stack([o, -wz, wy, wz, o, -wx, -wy, wx, o], -1).view(*w.shape[:-1], 3, 3)


def so3_exp(w, eps=1e-6):
    """Rodrigues: rotvec [..., 3] -> R [..., 3, 3]."""
    theta2 = (w * w).sum(-1, keepdim=True)
    theta = theta2.clamp(min=eps ** 2).sqrt()
    small = theta2 < eps ** 2
    # Taylor expansions near zero
    a = torch.where(small, 1 - theta2 / 6, torch.sin(theta) / theta)
    b = torch.where(small, 0.5 - theta2 / 24, (1 - torch.cos(theta)) / theta2.clamp(min=eps ** 2))
    W = skew(w)
    eye = torch.eye(3, dtype=w.dtype, device=w.device).expand(W.shape)
    return eye + a[..., None] * W + b[..., None] * (W @ W)


def so3_log(R, eps=1e-6):
    """R [..., 3, 3] -> rotvec [..., 3]. Not accurate for angles close to pi (fine for VO)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos = ((tr - 1) / 2).clamp(-1 + 1e-7, 1 - 1e-7)
    theta = torch.acos(cos)
    v = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                     R[..., 0, 2] - R[..., 2, 0],
                     R[..., 1, 0] - R[..., 0, 1]], -1)
    s = torch.sin(theta)
    factor = torch.where(theta < eps, 0.5 + theta ** 2 / 12, theta / (2 * s.clamp(min=eps)))
    return factor[..., None] * v


def pose_vec_to_mat(vec):
    """(rotvec, t) [..., 6] -> T [..., 4, 4]."""
    R = so3_exp(vec[..., :3])
    return make_T(R, vec[..., 3:])


def pose_mat_to_vec(T):
    return torch.cat([so3_log(T[..., :3, :3]), T[..., :3, 3]], -1)


def make_T(R, t):
    T = torch.zeros(*R.shape[:-2], 4, 4, dtype=R.dtype, device=R.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1
    return T


def inv_T(T):
    R = T[..., :3, :3].transpose(-1, -2)
    t = -(R @ T[..., :3, 3:])[..., 0]
    return make_T(R, t)


def transform_points(T, X):
    """T [B,4,4], X [B,3,N] or [B,3,H,W] -> same shape."""
    shape = X.shape
    X = X.reshape(shape[0], 3, -1)
    Y = T[:, :3, :3] @ X + T[:, :3, 3:]
    return Y.view(shape)


def rotation_angle(R):
    """Geodesic rotation angle (rad) of R [..., 3, 3]."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    return torch.acos(((tr - 1) / 2).clamp(-1, 1))
