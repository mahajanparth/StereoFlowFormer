"""Flow / disparity / segmentation / pose metrics."""
import numpy as np
import torch

from ..losses.pose import pose_error_vec


def flow_metrics(pred, gt, valid):
    """EPE and KITTI Fl (outlier: >3px and >5% of GT magnitude)."""
    v = valid[:, 0] > 0.5
    err = (pred - gt).norm(dim=1)
    mag = gt.norm(dim=1)
    out = ((err > 3) & (err > 0.05 * mag)).float()
    return dict(epe=err[v].mean().item(), fl=out[v].mean().item() * 100)


def disp_metrics(pred, gt, valid):
    """EPE and D1 (outlier: >3px and >5%)."""
    v = valid[:, 0] > 0.5
    err = (pred - gt).abs()[:, 0]
    out = ((err > 3) & (err > 0.05 * gt[:, 0])).float()
    return dict(epe=err[v].mean().item(), d1=out[v].mean().item() * 100)


def moving_iou(masks, moving, bg_slot=0):
    """IoU of predicted dynamic region (argmax != background) vs GT moving-object mask."""
    pred = masks.argmax(1) != bg_slot
    gt = moving[:, 0].bool()
    inter = (pred & gt).sum().item()
    union = (pred | gt).sum().item()
    return dict(mov_iou=inter / union if union else float('nan'),
                mov_precision=inter / max(pred.sum().item(), 1), mov_recall=inter / max(gt.sum().item(), 1))


def fg_ari(masks, inst, moving):
    """Adjusted Rand index over GT moving-object pixels (instance separation quality)."""
    from sklearn.metrics import adjusted_rand_score  # optional dependency
    scores = []
    for b in range(masks.shape[0]):
        m = moving[b, 0].bool()
        if m.sum() < 50:
            continue
        scores.append(adjusted_rand_score(inst[b, 0][m].cpu().numpy(), masks[b].argmax(0)[m].cpu().numpy()))
    return dict(fg_ari=float(np.mean(scores)) if scores else float('nan'))


def pose_metrics(pose_vec, T_gt):
    e = pose_error_vec(pose_vec, T_gt)
    return dict(rot_deg=(e[:, :3].norm(dim=1).mean() * 57.2958).item(), trans_m=e[:, 3:].norm(dim=1).mean().item())
