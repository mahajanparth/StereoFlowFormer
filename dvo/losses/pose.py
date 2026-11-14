"""Pose-head losses: supervised (VKITTI2 GT) and self-supervised (rigid flow on background)."""
import torch

from ..geometry.se3 import pose_vec_to_mat, so3_log
from ..geometry.warp import rigid_flow
from .motion_seg import charbonnier


def pose_error_vec(pose_vec, T_gt):
    """Residual (rotvec of R_gt^T R_pred, t_pred - t_gt) [B,6]."""
    T = pose_vec_to_mat(pose_vec.float())
    r = so3_log(T_gt[:, :3, :3].transpose(1, 2) @ T[:, :3, :3])
    return torch.cat([r, T[:, :3, 3] - T_gt[:, :3, 3]], 1)


def pose_supervised_loss(pose, logvar, T_gt, rot_weight=20.0):
    """Gaussian NLL with predicted per-dimension log-variance; rotation residual up-weighted (rad vs m)."""
    e = pose_error_vec(pose, T_gt.float())
    e = torch.cat([e[:, :3] * rot_weight, e[:, 3:]], 1)
    lv = logvar.float()
    nll = 0.5 * (e ** 2 * torch.exp(-lv) + lv)
    l1 = e.abs().mean()
    return nll.mean() + l1, dict(pose_rot_deg=(e[:, :3].norm(dim=1) / rot_weight).mean().detach() * 57.2958,
                                 pose_trans_m=e[:, 3:].norm(dim=1).mean().detach())


def pose_rigid_flow_loss(out, occ4=None):
    """Ego-motion rigid flow (pose head, with grad) must match network flow (detached) on background."""
    low = out['low']
    T = pose_vec_to_mat(out['pose'].float())
    rflow, ok = rigid_flow(low['depth_t'].detach(), low['K'], T)
    bg = out['masks4'][:, :1].detach().float()
    w = bg * ok.float() * (low['depth_t'] < 80).float()
    if occ4 is not None:
        w = w * occ4
    err = charbonnier((rflow - low['flow'].detach()).norm(dim=1, keepdim=True))
    return (err * w).sum() / w.sum().clamp(min=1)


def pose_kabsch_consistency(out, R0, t0):
    """Agreement between the pose head and the geometric background motion (Kabsch on slot 0)."""
    T = pose_vec_to_mat(out['pose'].float())
    r = so3_log(R0.transpose(1, 2) @ T[:, :3, :3])
    return (r.norm(dim=1) * 20 + (T[:, :3, 3] - t0).norm(dim=1)).mean()
