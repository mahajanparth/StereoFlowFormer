import torch

from dvo.geometry.se3 import pose_mat_to_vec, pose_vec_to_mat
from dvo.geometry.warp import backproject, rigid_flow
from dvo.losses.motion_seg import motion_seg_loss, pseudo_labels
from dvo.losses.pose import pose_rigid_flow_loss, pose_supervised_loss
from dvo.losses.supervised import sequence_loss


def synthetic_scene(B=1, h=40, w=120, K_slots=4):
    torch.manual_seed(0)
    K4 = torch.tensor([[180.0, 0, w / 2], [0, 180.0, h / 2], [0, 0, 1]]).expand(B, 3, 3).clone()
    depth = 5 + 20 * torch.rand(B, 1, h, w)
    obj = torch.zeros(B, 1, h, w, dtype=torch.bool)
    obj[..., 10:30, 40:80] = True
    T_ego = pose_vec_to_mat(torch.tensor([[0.0, 0.01, 0.0, 0.0, 0.0, -1.0]]).expand(B, 6))
    T_obj = pose_vec_to_mat(torch.tensor([[0.0, -0.03, 0.0, 1.5, 0.0, 1.5]]).expand(B, 6))
    f_ego, _ = rigid_flow(depth, K4, T_ego)
    f_obj, _ = rigid_flow(depth, K4, T_obj)
    flow = torch.where(obj, f_obj, f_ego)
    X_t = backproject(depth, K4)
    T_pix = torch.where(obj[..., None, None].squeeze(1), T_obj[:, None, None], T_ego[:, None, None])  # [B,h,w,4,4]
    X1 = (T_pix[..., :3, :3] @ X_t.permute(0, 2, 3, 1)[..., None])[..., 0] + T_pix[..., :3, 3]
    low = dict(flow=flow, depth_t=depth, disp_t=180.0 * 0.54 / depth, K=K4, X_t=X_t, X_t1_at=X1.permute(0, 3, 1, 2),
               inb=torch.ones(B, 1, h, w))
    return low, obj, T_ego


def _masks(obj, K_slots, good=True):
    B, _, h, w = obj.shape
    logits = torch.full((B, K_slots, h, w), -10.0)
    logits[:, 0] = 10.0
    if good:
        logits[:, 1][obj[:, 0]] = 20.0
    return logits


def test_pseudo_labels_find_moving_object_and_ego_motion():
    low, obj, T_ego = synthetic_scene()
    labels, (R, t), _ = pseudo_labels(low, torch.ones_like(low['depth_t']), num_slots=4)
    assert torch.allclose(t, T_ego[:, :3, 3], atol=1e-2)
    inst = labels[0] == 1
    assert (inst & obj[0, 0]).sum() > 0.8 * obj.sum()          # object recovered as instance 1
    bg = labels[0][~obj[0, 0]]
    assert (bg > 0).float().mean() < 0.01                       # background never labelled moving
    assert (bg == 0).float().mean() > 0.6                       # mostly static (rest: stereo band/border/ring ignored)
    assert labels.max() == 1


def test_motion_seg_prefers_correct_segmentation_and_is_permutation_invariant():
    low, obj, T_ego = synthetic_scene()
    img = torch.rand(1, 3, *obj.shape[-2:])
    losses = {}
    for name, (good, slot) in {'good': (True, 1), 'good_other_slot': (True, 3), 'bad': (False, 1)}.items():
        logits = torch.full((1, 4, *obj.shape[-2:]), -10.0)
        logits[:, 0] = 10.0
        if good:
            logits[:, slot][obj[:, 0]] = 20.0
        logits.requires_grad_(True)
        out = dict(mask_logits4=logits, masks4=logits.softmax(1), low=low)
        total, stats, _ = motion_seg_loss(out, img)
        total.backward()
        assert torch.isfinite(logits.grad).all()
        losses[name] = total.item()
    assert losses['good'] < 0.1 * losses['bad'], losses
    assert abs(losses['good'] - losses['good_other_slot']) < 1e-3, losses


def test_pose_losses_zero_at_gt():
    low, obj, T_ego = synthetic_scene()
    pv = pose_mat_to_vec(T_ego).requires_grad_(True)
    l, stats = pose_supervised_loss(pv, torch.zeros(1, 6), T_ego)
    assert stats['pose_trans_m'] < 1e-5 and stats['pose_rot_deg'] < 1e-3
    masks = _masks(obj, 4).softmax(1)
    out = dict(masks4=masks, low=low, pose=pv)
    assert pose_rigid_flow_loss(out) < 0.2  # only the charbonnier floor on background pixels


def test_sequence_loss_weights_last_pred():
    gt = torch.randn(2, 2, 8, 8)
    valid = torch.ones(2, 1, 8, 8)
    l_good, epe = sequence_loss([gt + 1, gt], gt, valid)
    l_bad, _ = sequence_loss([gt, gt + 1], gt, valid)
    assert epe < 1e-6 and l_good < l_bad
