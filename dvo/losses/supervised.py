"""Sequence losses over intermediate predictions (RAFT / UniMatch style)."""
import torch


def sequence_loss(preds, gt, valid, gamma=0.9, max_mag=400.0):
    """preds: list of [B,C,H,W]; gt [B,C,H,W]; valid [B,1,H,W]. Later predictions weigh more."""
    mag = gt.norm(dim=1, keepdim=True)
    valid = (valid > 0.5) & (mag < max_mag)
    n = len(preds)
    loss = 0.0
    denom = valid.float().sum().clamp(min=1)
    for i, p in enumerate(preds):
        w = gamma ** (n - i - 1)
        loss = loss + w * ((p.float() - gt).abs().sum(1, keepdim=True) * valid).sum() / denom
    epe = ((preds[-1].float() - gt).norm(dim=1, keepdim=True)[valid]).mean() if valid.any() else gt.new_zeros(())
    return loss, epe.detach()
