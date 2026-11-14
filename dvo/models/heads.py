"""Mask (motion-instance) head and ego-motion pose head.

Both consume a shared 1/4-resolution context map built from
  * flow-branch transformer feature of L_t       (temporal matching context)
  * stereo-branch transformer feature of L_t     (stereo / geometry context)
  * geometric encoding (detached): flow / fx, inverse depth (1/m), normalised rays K^-1 p and
    3D scene flow (X_t+1 - X_t, metres) obtained from flow + both disparities.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unimatch.position import PositionEmbeddingSine

GEO_CHANNELS = 8


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c)
        self.norm2 = nn.GroupNorm(8, c)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        return F.gelu(x + self.norm2(self.conv2(y)))


class ContextEncoder(nn.Module):
    """Fuses branch features + geometry at 1/4 res; also produces 1/16-res tokens."""

    def __init__(self, in_feat=256, dim=128, token_dim=192):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv2d(in_feat + GEO_CHANNELS, dim, 3, padding=1), nn.GroupNorm(8, dim), nn.GELU())
        self.blocks = nn.Sequential(ResBlock(dim), ResBlock(dim))
        self.down = nn.Sequential(
            nn.Conv2d(dim, token_dim, 3, stride=2, padding=1), nn.GroupNorm(8, token_dim), nn.GELU(),
            nn.Conv2d(token_dim, token_dim, 3, stride=2, padding=1), nn.GroupNorm(8, token_dim), nn.GELU())
        self.pos = PositionEmbeddingSine(num_pos_feats=token_dim // 2)

    def forward(self, feat, geo):
        x = self.blocks(self.inp(torch.cat([feat, geo], 1)))      # [B,dim,h,w] (1/4)
        t = self.down(x)                                            # [B,tok,h/4,w/4] (1/16)
        pos = self.pos(t)
        return x, t.flatten(2).transpose(1, 2), pos.flatten(2).transpose(1, 2)


class QueryDecoderLayer(nn.Module):
    def __init__(self, d, nhead=8, ffn=4):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(d, d * ffn), nn.GELU(), nn.Linear(d * ffn, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, q, qpos, mem, mpos):
        h = self.n1(q)
        q = q + self.self_attn(h + qpos, h + qpos, h)[0]
        h = self.n2(q)
        q = q + self.cross_attn(h + qpos, mem + mpos, mem)[0]
        return q + self.ffn(self.n3(q))


class MaskHead(nn.Module):
    """Mask2Former-lite: K slot queries -> K soft masks (softmax over slots). Slot 0 = background."""

    def __init__(self, num_slots=8, pix_dim=128, token_dim=192, num_layers=3):
        super().__init__()
        self.num_slots = num_slots
        self.query = nn.Parameter(torch.randn(num_slots, token_dim))  # diverse init breaks slot symmetry
        self.query_pos = nn.Parameter(torch.randn(num_slots, token_dim) * 0.02)
        self.layers = nn.ModuleList([QueryDecoderLayer(token_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(token_dim)
        self.mask_embed = nn.Sequential(nn.Linear(token_dim, token_dim), nn.GELU(), nn.Linear(token_dim, pix_dim))
        self.pix_proj = nn.Conv2d(pix_dim, pix_dim, 1)

    def forward(self, pix, tokens, tok_pos):
        b = pix.shape[0]
        q = self.query[None].expand(b, -1, -1)
        qpos = self.query_pos[None].expand(b, -1, -1)
        for layer in self.layers:
            q = layer(q, qpos, tokens, tok_pos)
        emb = self.mask_embed(self.norm(q))                         # [B,K,C]
        logits = torch.einsum('bkc,bchw->bkhw', emb, self.pix_proj(pix)) / math.sqrt(emb.shape[-1])
        return logits                                                # [B,K,h,w] at 1/4


class PoseHead(nn.Module):
    """Pose token attends to 1/16 tokens (+ background-mask channel) -> (rotvec, t) and log-variance."""

    def __init__(self, token_dim=192, num_layers=4, nhead=8):
        super().__init__()
        self.mask_proj = nn.Linear(1, token_dim)
        self.token = nn.Parameter(torch.zeros(1, 1, token_dim))
        layer = nn.TransformerEncoderLayer(token_dim, nhead, token_dim * 4, dropout=0.0, activation='gelu',
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(token_dim)
        self.out = nn.Linear(token_dim, 12)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.rot_scale, self.trans_scale = 0.05, 1.0   # typical inter-frame motion (rad, m)

    def forward(self, tokens, tok_pos, bg_mask_tokens):
        x = tokens + tok_pos + self.mask_proj(bg_mask_tokens)
        x = torch.cat([self.token.expand(x.shape[0], -1, -1), x], 1)
        y = self.out(self.norm(self.encoder(x)[:, 0]))
        pose = torch.cat([y[:, :3] * self.rot_scale, y[:, 3:6] * self.trans_scale], 1)
        logvar = y[:, 6:].clamp(-10, 10)
        return pose, logvar
