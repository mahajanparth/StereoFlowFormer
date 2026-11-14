"""Dense robust PnP: estimate T minimising sum_p w_p rho(|| pi(T X_p) - q_p ||) by batched Gauss-Newton.

X_p: 3D points of frame t (from its depth), q_p: their correspondences in frame t+1 (p + flow).
Using 2D reprojection error (not 3D Kabsch error) keeps far-point depth noise from dominating.
"""
import torch

from .se3 import so3_exp


def dense_pnp(X, q, K, w, R=None, t=None, iters=10, sigma=1.0, damping=1e-4, hard_inlier_final=True):
    """X [B,3,N], q [B,2,N] pixels, K [B,3,3], w [B,N] prior weights (0 = unusable).

    Returns R [B,3,3], t [B,3], residual norm [B,N]. Runs in float64 for stability.
    """
    dtype = X.dtype
    X, q, K, w = X.double(), q.double(), K.double(), w.double()
    B, _, N = X.shape
    R = torch.eye(3, dtype=X.dtype, device=X.device).expand(B, 3, 3).clone() if R is None else R.double()
    t = torch.zeros(B, 3, dtype=X.dtype, device=X.device) if t is None else t.double()
    fx, fy, cx, cy = K[:, 0, 0, None], K[:, 1, 1, None], K[:, 0, 2, None], K[:, 1, 2, None]
    rw = w
    for it in range(iters):
        Y = R @ X + t[..., None]                                    # [B,3,N]
        x, y, z = Y[:, 0], Y[:, 1], Y[:, 2].clamp(min=1e-3)
        u, v = fx * x / z + cx, fy * y / z + cy
        e = torch.stack([u - q[:, 0], v - q[:, 1]], -1)            # [B,N,2]
        rn = e.norm(dim=-1)
        if it > 0:
            rw = w / (1 + (rn / sigma) ** 2)                        # Cauchy IRLS
            if hard_inlier_final and it == iters - 1:
                rw = w * (rn < 2 * sigma).double()
        rw = rw * (Y[:, 2] > 1e-2).double()
        # Jacobian of projection wrt left perturbation (v, omega): dY = v + omega x Y
        zi = 1 / z
        o = torch.zeros_like(x)
        Jp = torch.stack([torch.stack([fx * zi, o, -fx * x * zi ** 2], -1),
                          torch.stack([o, fy * zi, -fy * y * zi ** 2], -1)], -2)   # [B,N,2,3]
        Yt = Y.transpose(1, 2)                                      # [B,N,3]
        Yx = torch.zeros(B, N, 3, 3, dtype=X.dtype, device=X.device)
        Yx[..., 0, 1], Yx[..., 0, 2] = -Yt[..., 2], Yt[..., 1]
        Yx[..., 1, 0], Yx[..., 1, 2] = Yt[..., 2], -Yt[..., 0]
        Yx[..., 2, 0], Yx[..., 2, 1] = -Yt[..., 1], Yt[..., 0]
        J = torch.cat([Jp, -Jp @ Yx], -1)                           # [B,N,2,6]
        Jw = J * rw[..., None, None]
        H = torch.einsum('bnij,bnik->bjk', Jw, J)
        g = torch.einsum('bnij,bni->bj', Jw, e)
        H = H + damping * torch.diag_embed(torch.diagonal(H, dim1=-2, dim2=-1)) + 1e-9 * torch.eye(6, dtype=H.dtype, device=H.device)
        xi = -torch.linalg.solve(H, g)                              # [B,6]
        dR = so3_exp(xi[:, 3:])
        R = dR @ R
        t = (dR @ t[..., None])[..., 0] + xi[:, :3]
    return R.to(dtype), t.to(dtype), rn.to(dtype)
