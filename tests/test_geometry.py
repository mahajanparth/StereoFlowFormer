import math

import torch

from dvo.geometry.camera import depth_to_disp, disp_to_depth
from dvo.geometry.procrustes import weighted_kabsch
from dvo.geometry.se3 import inv_T, pose_mat_to_vec, pose_vec_to_mat, rotation_angle, so3_exp, so3_log
from dvo.geometry.warp import backproject, coords_grid, disp_warp, fb_consistency, flow_warp, project, rigid_flow

torch.manual_seed(0)


def make_K(b=2, f=100.0, w=64, h=48):
    K = torch.tensor([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    return K.expand(b, 3, 3).clone()


def test_so3_roundtrip():
    w = torch.randn(16, 3) * 0.8
    assert torch.allclose(so3_log(so3_exp(w)), w, atol=1e-5)
    R = so3_exp(w)
    assert torch.allclose(R @ R.transpose(-1, -2), torch.eye(3).expand(16, 3, 3), atol=1e-5)
    assert torch.allclose(so3_exp(torch.zeros(1, 3)), torch.eye(3)[None])


def test_pose_vec_roundtrip_and_inverse():
    v = torch.randn(8, 6) * 0.5
    T = pose_vec_to_mat(v)
    assert torch.allclose(pose_mat_to_vec(T), v, atol=1e-5)
    assert torch.allclose(T @ inv_T(T), torch.eye(4).expand(8, 4, 4), atol=1e-5)


def test_backproject_project_roundtrip():
    K = make_K()
    depth = torch.rand(2, 1, 48, 64) * 10 + 1
    p, valid = project(backproject(depth, K), K)
    assert valid.all()
    assert torch.allclose(p, coords_grid(2, 48, 64), atol=1e-3)


def test_rigid_flow_identity_and_translation():
    K = make_K()
    depth = torch.full((2, 1, 48, 64), 5.0)
    flow, _ = rigid_flow(depth, K, torch.eye(4).expand(2, 4, 4))
    assert flow.abs().max() < 1e-4
    T = torch.eye(4).repeat(2, 1, 1)
    T[:, 0, 3] = 0.5  # move points +x by 0.5m at depth 5 -> 100*0.5/5 = 10 px
    flow, _ = rigid_flow(depth, K, T)
    assert torch.allclose(flow[:, 0], torch.full_like(flow[:, 0], 10.0), atol=1e-3)
    assert flow[:, 1].abs().max() < 1e-4


def test_flow_and_disp_warp():
    img = torch.rand(1, 3, 48, 64)
    flow = torch.zeros(1, 2, 48, 64)
    flow[:, 0] = 3.0
    shifted = torch.roll(img, shifts=-3, dims=-1)        # shifted[x] = img[x + 3]
    warped, inb = flow_warp(img, flow)
    assert torch.allclose(warped[..., :-3], shifted[..., :-3], atol=1e-5)
    disp = torch.full((1, 1, 48, 64), 4.0)              # left x sees right x - 4
    right = torch.roll(img, shifts=-4, dims=-1)         # right[x] = img[x + 4]
    warped, _ = disp_warp(right, disp)
    assert torch.allclose(warped[..., 4:], img[..., 4:], atol=1e-5)


def test_fb_consistency_constant_flow():
    fw = torch.zeros(1, 2, 48, 64); fw[:, 0] = 2.0
    bw = -fw
    occ = fb_consistency(fw, bw)
    assert occ[..., :-3].min() == 1


def test_disp_depth():
    d = torch.rand(3, 1, 4, 4) * 50 + 1
    fx, B = torch.tensor([700.0, 720.0, 710.0]), 0.54
    assert torch.allclose(depth_to_disp(disp_to_depth(d, fx, B), fx, B), d, atol=1e-4)


def test_weighted_kabsch_recovers_motion_with_outliers():
    B, N = 4, 500
    R = so3_exp(torch.randn(B, 3) * 0.3)
    t = torch.randn(B, 3)
    X = torch.randn(B, N, 3) * 5
    Y = (R @ X.transpose(1, 2)).transpose(1, 2) + t[:, None]
    w = torch.ones(B, N)
    Y[:, :100] += torch.randn(B, 100, 3) * 10            # outliers with zero weight
    w[:, :100] = 0
    R_est, t_est = weighted_kabsch(X, Y, w)
    assert rotation_angle(R_est.transpose(-1, -2) @ R).max() < 1e-4
    assert torch.allclose(t_est, t, atol=1e-4)


def test_weighted_kabsch_gradients_finite():
    X = torch.randn(2, 200, 3)
    Y = X + 0.1
    w = torch.rand(2, 200, requires_grad=True)
    R, t = weighted_kabsch(X, Y, w)
    (R.sum() + t.sum()).backward()
    assert torch.isfinite(w.grad).all()
