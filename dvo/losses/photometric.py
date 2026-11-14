"""Self-supervised photometric losses for flow and stereo (KITTI stage)."""
import torch
import torch.nn.functional as F

from ..geometry.warp import disp_warp, fb_consistency, flow_warp


def ssim(x, y, C1=0.01 ** 2, C2=0.03 ** 2):
    mu_x = F.avg_pool2d(x, 3, 1, 1)
    mu_y = F.avg_pool2d(y, 3, 1, 1)
    sx = F.avg_pool2d(x * x, 3, 1, 1) - mu_x ** 2
    sy = F.avg_pool2d(y * y, 3, 1, 1) - mu_y ** 2
    sxy = F.avg_pool2d(x * y, 3, 1, 1) - mu_x * mu_y
    n = (2 * mu_x * mu_y + C1) * (2 * sxy + C2)
    d = (mu_x ** 2 + mu_y ** 2 + C1) * (sx + sy + C2)
    return ((1 - n / d) / 2).clamp(0, 1)


def photo_error(a, b, alpha=0.85):
    """Per-pixel SSIM + L1 error [B,1,H,W]."""
    return alpha * ssim(a, b).mean(1, keepdim=True) + (1 - alpha) * (a - b).abs().mean(1, keepdim=True)


def masked_mean(x, m):
    return (x * m).sum() / m.sum().clamp(min=1)


def smoothness(field, img, order=1):
    """Edge-aware smoothness of a dense field, normalised by its mean magnitude."""
    field = field / (field.abs().mean((2, 3), keepdim=True) + 1e-3)
    gx_i = (img[..., :, 1:] - img[..., :, :-1]).abs().mean(1, keepdim=True)
    gy_i = (img[..., 1:, :] - img[..., :-1, :]).abs().mean(1, keepdim=True)
    gx = field[..., :, 1:] - field[..., :, :-1]
    gy = field[..., 1:, :] - field[..., :-1, :]
    if order == 2:
        gx, gy = gx[..., :, 1:] - gx[..., :, :-1], gy[..., 1:, :] - gy[..., :-1, :]
        gx_i, gy_i = gx_i[..., :, 1:], gy_i[..., 1:, :]
    return (gx.abs() * torch.exp(-10 * gx_i)).mean() + (gy.abs() * torch.exp(-10 * gy_i)).mean()


def flow_photometric_loss(Lt, Lt1, flow_fw, flow_bw):
    """Bidirectional photometric loss with forward-backward occlusion masking. Returns loss, occ masks."""
    occ_fw = fb_consistency(flow_fw, flow_bw).detach()
    occ_bw = fb_consistency(flow_bw, flow_fw).detach()
    w_fw, inb_fw = flow_warp(Lt1, flow_fw)
    w_bw, inb_bw = flow_warp(Lt, flow_bw)
    m_fw, m_bw = occ_fw * inb_fw, occ_bw * inb_bw
    loss = masked_mean(photo_error(Lt, w_fw), m_fw) + masked_mean(photo_error(Lt1, w_bw), m_bw)
    return loss, m_fw, m_bw


def stereo_photometric_loss(L, R, disp):
    """Right->left warp with left disparity; pixels whose match falls outside the right image are masked."""
    warped, inb = disp_warp(R, disp)
    return masked_mean(photo_error(L, warped), inb.float())
