"""Self-supervised motion-instance segmentation from flow + stereo depth (no mask labels).

Pseudo-labels are derived per batch from the network's own geometry, at 1/4 resolution:
  1. robust ego-motion: dense PnP (robust Gauss-Newton on 2D reprojection error) between 3D points of frame t
     (stereo depth) and their flow correspondences in t+1. The static background dominates driving scenes,
     so this converges to the camera motion without any pose supervision.
  2. motion evidence: residual r = || flow - rigidflow(T_ego) ||. r above a high threshold -> moving,
     below a low threshold -> static, in between / far / occluded -> ignored.
  3. instances: connected components of the (cleaned) moving region, largest K-1 kept.
The mask head is trained with a permutation-invariant loss: predicted slots 1..K-1 are Hungarian-matched
to the pseudo-instances (cost = CE + Dice, as in Mask2Former); slot 0 is always the static background.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

from ..geometry.dense_pnp import dense_pnp
from ..geometry.procrustes import weighted_kabsch
from ..geometry.warp import backproject, coords_grid, project
from .photometric import smoothness


def slot_motions(masks, X_t, X_t1, valid, depth, max_depth=40.0):
    """masks [B,K,h,w], X_t/X_t1 [B,3,h,w], valid [B,1,h,w] -> R [B,K,3,3], t [B,K,3]."""
    B, K = masks.shape[:2]
    rel = valid * (depth < max_depth).float() / (1 + depth / 10.0)    # near points are more reliable
    w = (masks.detach() * rel).flatten(2)                                # [B,K,N]
    X = X_t.flatten(2).transpose(1, 2)[:, None].expand(B, K, -1, 3)
    Y = X_t1.flatten(2).transpose(1, 2)[:, None].expand(B, K, -1, 3)
    return weighted_kabsch(X, Y, w)


def rigid_flows(X_t, K4, R, t):
    """Rigid flow for every slot: [B,K,2,h,w] and validity [B,K,1,h,w]."""
    B, Kn = R.shape[:2]
    h, w = X_t.shape[-2:]
    X = X_t.flatten(2)                                                   # [B,3,N]
    Y = R @ X[:, None] + t[..., None]                                    # [B,K,3,N]
    Y = Y.reshape(B * Kn, 3, h, w)
    p, ok = project(Y, K4.repeat_interleave(Kn, 0))
    grid = coords_grid(B * Kn, h, w, X_t.device, X_t.dtype)
    return (p - grid).view(B, Kn, 2, h, w), ok.view(B, Kn, 1, h, w).float()


def charbonnier(x, eps=0.01, q=0.4):
    return (x ** 2 + eps ** 2) ** q


@torch.no_grad()
def robust_ego_motion(low, valid, iters=12, sigma=0.5, max_depth=60.0):
    """Dense robust PnP for the dominant (camera) motion: 3D points of frame t (stereo depth) vs their flow
    correspondences in frame t+1. Returns R [B,3,3], t [B,3], residual [B,1,h,w] (1/4-res px), rigid flow."""
    X_t, depth, flow, K4 = low['X_t'], low['depth_t'], low['flow'], low['K']
    B, _, h, w = depth.shape
    q = coords_grid(B, h, w, depth.device, depth.dtype) + flow
    wgt = (valid * (depth < max_depth).float()).flatten(1)
    R, t, _ = dense_pnp(X_t.flatten(2), q.flatten(2), K4, wgt, iters=iters, sigma=sigma)
    rflow, ok = rigid_flows(X_t, K4, R[:, None], t[:, None])
    r = (flow - rflow[:, 0]).norm(dim=1, keepdim=True)
    return R, t, r, rflow[:, 0]


@torch.no_grad()
def pseudo_labels(low, valid, num_slots, tau_lo=0.4, tau_hi=1.0, rel_lo=0.04, rel_hi=0.08, k_lo=1.0, k_hi=2.0,
                  disp_sigma=0.25, max_depth=40.0, min_size=24, border=2):
    """Returns labels [B,h,w] long (-1 ignore, 0 static, 1..n instance), ego (R, t), residual.

    Thresholds are uncertainty-aware: on top of absolute/relative terms they add the change of the ego rigid
    flow caused by a `disp_sigma` (1/4-res px) disparity error, so near / low-disparity pixels need more evidence.
    """
    R, t, r, rflow = robust_ego_motion(low, valid)
    depth, disp = low['depth_t'], low['disp_t']
    mag = rflow.norm(dim=1, keepdim=True)
    depth_pert = depth * disp / (disp - disp_sigma).clamp(min=1e-2)        # depth if disparity were 1 sigma smaller
    rflow_pert, _ = rigid_flows(backproject(depth_pert, low['K']), low['K'], R[:, None], t[:, None])
    unc = (rflow_pert[:, 0] - rflow).norm(dim=1, keepdim=True)
    B, _, h, w = depth.shape
    x = coords_grid(B, h, w, depth.device, depth.dtype)[:, :1]
    stereo_ok = x - disp >= 1                                                # stereo match inside the right image
    inner = torch.zeros_like(stereo_ok)
    inner[..., border:h - border, border:w - border] = True
    near = depth < max_depth
    ok = (valid > 0.5) & near & stereo_ok & inner
    hi = torch.maximum(torch.full_like(mag, tau_hi), rel_hi * mag) + k_hi * unc
    lo = torch.maximum(torch.full_like(mag, tau_lo), rel_lo * mag) + k_lo * unc
    moving = (r > hi) & ok
    static = (r < lo) & ok
    # majority filter against speckle; erode to a confident core and ignore the boundary ring, where flow and
    # disparity bleed across object edges (the mask head learns exact boundaries from appearance)
    moving = F.avg_pool2d(moving.float(), 3, 1, 1) > 0.5
    core = -F.max_pool2d(-moving.float(), 3, 1, 1) > 0.5
    ring = (F.max_pool2d(moving.float(), 5, 1, 2) > 0.5) & ~core
    labels = torch.full_like(r[:, 0], -1, dtype=torch.long)
    labels[static[:, 0]] = 0
    labels[(~near & (valid > 0.5) & stereo_ok & inner)[:, 0]] = 0   # far: motion unobservable, assume static
    labels[ring[:, 0]] = -1
    moving = core
    mv = moving[:, 0].cpu().numpy()
    inst = np.zeros(mv.shape, np.int64)
    for b in range(len(mv)):
        lab, n = ndimage.label(mv[b], structure=np.ones((3, 3)))
        if n == 0:
            continue
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        order = np.argsort(-sizes)
        k = 1
        for idx in order:
            if sizes[idx] < min_size or k >= num_slots:
                break
            inst[b][lab == idx + 1] = k
            k += 1
    inst = torch.from_numpy(inst).to(labels.device)
    labels = torch.where(inst > 0, inst, labels)
    return labels, (R, t), r


def matched_seg_loss(logits, labels, fg_weight=5.0):
    """Hungarian-matched CE + Dice. logits [B,K,h,w] (float32), labels [B,h,w] (-1/0/1..n)."""
    B, K = logits.shape[:2]
    logp = logits.log_softmax(1)
    prob = logp.exp()
    target = labels.clone()
    dice_terms = []
    for b in range(B):
        n = int(labels[b].max().item())
        if n <= 0:
            continue
        M = torch.stack([(labels[b] == i + 1).flatten().float() for i in range(n)])   # [n,N]
        P = prob[b, 1:].flatten(1)                                                     # [K-1,N]
        LP = logp[b, 1:].flatten(1)
        area = M.sum(1, keepdim=True).clamp(min=1)
        ce_cost = -(M @ LP.T) / area                                                   # [n,K-1]
        dice_cost = 1 - 2 * (M @ P.T) / (area + P.sum(1)[None])
        rows, cols = linear_sum_assignment((ce_cost + dice_cost).detach().cpu().numpy())
        tb = target[b]
        for i, k in zip(rows, cols):
            tb[labels[b] == i + 1] = k + 1
            dice_terms.append(dice_cost[i, k])
    valid = target >= 0
    weight = torch.where(target > 0, fg_weight, 1.0) * valid
    ce = F.nll_loss(logp, target.clamp(min=0), reduction='none')
    l_ce = (ce * weight).sum() / weight.sum().clamp(min=1)
    l_dice = torch.stack(dice_terms).mean() if dice_terms else logits.new_zeros(())
    return l_ce, l_dice, target


def motion_seg_loss(out, img4, occ4=None, w_dice=1.0, w_smooth=0.05, **label_kw):
    """out: model outputs (needs mask_logits4, masks4, low{...}). img4: L_t at 1/4 res."""
    low = out['low']
    logits = out['mask_logits4'].float()
    valid = low['inb'].float()
    if occ4 is not None:
        valid = valid * occ4
    labels, (R, t), r = pseudo_labels(low, valid, logits.shape[1], **label_kw)
    l_ce, l_dice, target = matched_seg_loss(logits, labels)
    l_smooth = smoothness(out['masks4'].float(), img4)
    total = l_ce + w_dice * l_dice + w_smooth * l_smooth
    masks = out['masks4'].float()
    stats = dict(seg_ce=l_ce.detach(), seg_dice=l_dice.detach(),
                 seg_pseudo_inst=labels.amax((1, 2)).clamp(min=0).float().mean(),
                 seg_pseudo_moving=(labels > 0).float().mean(), seg_ignored=(labels < 0).float().mean(),
                 seg_pred_moving=(masks.argmax(1) > 0).float().mean())
    return total, stats, (R, t)
