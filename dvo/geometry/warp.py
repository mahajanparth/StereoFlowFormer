"""Back-projection, projection, rigid flow and image warping (torch, batched)."""
import torch
import torch.nn.functional as F

from .se3 import transform_points


def coords_grid(b, h, w, device=None, dtype=torch.float32):
    """Pixel coordinates [B, 2, H, W] with channel order (x, y)."""
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                          torch.arange(w, device=device, dtype=dtype), indexing='ij')
    return torch.stack([x, y], 0)[None].expand(b, -1, -1, -1)


def backproject(depth, K):
    """depth [B,1,H,W], K [B,3,3] -> points in camera frame [B,3,H,W]."""
    b, _, h, w = depth.shape
    pix = coords_grid(b, h, w, depth.device, depth.dtype)
    ones = torch.ones_like(pix[:, :1])
    pix_h = torch.cat([pix, ones], 1).view(b, 3, -1)
    rays = torch.linalg.solve(K.to(depth.dtype), pix_h)  # K^-1 p
    return (rays * depth.view(b, 1, -1)).view(b, 3, h, w)


def project(X, K, eps=1e-3):
    """X [B,3,H,W] -> pixels [B,2,H,W], plus positive-depth mask [B,1,H,W]."""
    b, _, h, w = X.shape
    p = (K.to(X.dtype) @ X.view(b, 3, -1)).view(b, 3, h, w)
    z = p[:, 2:3]
    valid = z > eps
    return p[:, :2] / z.clamp(min=eps), valid


def rigid_flow(depth, K, T):
    """Flow induced by camera/object motion T (X' = T X) for a depth map. Returns flow, valid."""
    X = backproject(depth, K)
    X2 = transform_points(T, X)
    p2, valid = project(X2, K)
    b, _, h, w = depth.shape
    return p2 - coords_grid(b, h, w, depth.device, depth.dtype), valid


def sample(img, coords, mode='bilinear', padding_mode='zeros'):
    """Sample img [B,C,H,W] at pixel coords [B,2,H',W']. Returns samples and in-bounds mask."""
    h, w = img.shape[-2:]
    x = 2 * coords[:, 0] / max(w - 1, 1) - 1
    y = 2 * coords[:, 1] / max(h - 1, 1) - 1
    grid = torch.stack([x, y], -1)
    out = F.grid_sample(img, grid.to(img.dtype), mode=mode, padding_mode=padding_mode, align_corners=True)
    inb = ((x.abs() <= 1) & (y.abs() <= 1)).unsqueeze(1)
    return out, inb


def flow_warp(img, flow, **kw):
    """Backward-warp img (target frame) to the source frame using flow source->target."""
    b, _, h, w = flow.shape
    return sample(img, coords_grid(b, h, w, flow.device, flow.dtype) + flow, **kw)


def disp_warp(img_right, disp):
    """Warp the right image to the left view with left disparity (x_r = x_l - d)."""
    flow = torch.cat([-disp, torch.zeros_like(disp)], 1)
    return flow_warp(img_right, flow)


def fb_consistency(flow_fw, flow_bw, alpha=0.01, beta=0.5):
    """Forward-backward occlusion check (UnFlow). Returns non-occluded mask for fw flow [B,1,H,W]."""
    bw_warped, _ = flow_warp(flow_bw, flow_fw)
    diff = (flow_fw + bw_warped).pow(2).sum(1, keepdim=True)
    mag = flow_fw.pow(2).sum(1, keepdim=True) + bw_warped.pow(2).sum(1, keepdim=True)
    return (diff < alpha * mag + beta).to(flow_fw.dtype)
