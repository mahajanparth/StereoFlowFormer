"""Weighted rigid alignment (Kabsch / Umeyama without scale), batched and differentiable."""
import torch


def weighted_kabsch(X, Y, w, eps=1e-6):
    """Find R, t minimising sum_i w_i ||R X_i + t - Y_i||^2.

    X, Y: [..., N, 3]; w: [..., N] (non-negative). Returns R [..., 3, 3], t [..., 3].
    Computation runs in float64 for SVD stability; gradients flow through torch.linalg.svd.
    """
    dtype = X.dtype
    X, Y, w = X.double(), Y.double(), w.double()
    wsum = w.sum(-1, keepdim=True).clamp(min=eps)
    wn = (w / wsum).unsqueeze(-1)                       # [..., N, 1]
    mx = (wn * X).sum(-2, keepdim=True)
    my = (wn * Y).sum(-2, keepdim=True)
    Xc, Yc = X - mx, Y - my
    H = (wn * Xc).transpose(-1, -2) @ Yc                # [..., 3, 3]
    H = H + eps * torch.eye(3, dtype=H.dtype, device=H.device)  # avoid degenerate SVD grads
    U, S, Vh = torch.linalg.svd(H)
    V = Vh.transpose(-1, -2)
    d = torch.sign(torch.linalg.det(V @ U.transpose(-1, -2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = V @ D @ U.transpose(-1, -2)
    t = my.squeeze(-2) - (R @ mx.transpose(-1, -2)).squeeze(-1)
    return R.to(dtype), t.to(dtype)
