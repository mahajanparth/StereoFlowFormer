"""StereoFlowFormer: two stereo pairs -> flow, disparity, motion-instance masks, ego-motion."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..geometry.camera import disp_to_depth, scale_intrinsics
from ..geometry.warp import backproject, coords_grid, sample
from .heads import ContextEncoder, MaskHead, PoseHead
from .match_branch import MatchBranch, load_unimatch_weights

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class StereoFlowFormer(nn.Module):
    def __init__(self, num_slots=8, share_backbone=False, detach_geometry=True, freeze_branches=False):
        super().__init__()
        self.freeze_branches = freeze_branches
        self.flow = MatchBranch('flow')
        self.stereo = MatchBranch('stereo')
        if share_backbone:
            self.stereo.backbone = self.flow.backbone
            self.stereo.transformer = self.flow.transformer
        self.context = ContextEncoder(in_feat=256)
        self.mask_head = MaskHead(num_slots=num_slots)
        self.pose_head = PoseHead()
        self.detach_geometry = detach_geometry
        self.register_buffer('mean', torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def load_pretrained(self, flow_ckpt, stereo_ckpt):
        load_unimatch_weights(self.flow, flow_ckpt)
        if self.stereo.backbone is self.flow.backbone:
            sd = torch.load(stereo_ckpt, map_location='cpu', weights_only=False)['model']
            sd = {k: v for k, v in sd.items() if not k.startswith(('backbone.', 'transformer.'))}
            self.stereo.load_state_dict(sd, strict=False)
        else:
            load_unimatch_weights(self.stereo, stereo_ckpt)

    def set_freeze_branches(self, freeze):
        self.freeze_branches = freeze
        for br in (self.flow, self.stereo):
            br.requires_grad_(not freeze)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_branches:  # frozen branches run in eval mode (final estimates only)
            self.flow.eval()
            self.stereo.eval()
        return self

    def normalize(self, img):
        return (img - self.mean) / self.std

    def forward(self, Lt, Rt, Lt1, Rt1, K, baseline, run_heads=True):
        """Images in [0,1], [B,3,H,W]; H, W multiples of 32 (see `pad_inputs`).

        K: [B,3,3] intrinsics of the rectified left camera, baseline: [B] metres.
        """
        B, _, H, W = Lt.shape
        Lt, Rt, Lt1, Rt1 = map(self.normalize, (Lt, Rt, Lt1, Rt1))

        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_branches):
            fo = self.flow(Lt, Lt1, bidir=True)
            so = self.stereo(torch.cat([Lt, Lt1]), torch.cat([Rt, Rt1]))

        out = dict(
            flow_preds=fo['preds'],                        # each [2B,2,H,W]: fw (L_t->L_t+1) then bw
            flow_fw=fo['final'][:B], flow_bw=fo['final'][B:],
            disp_preds=so['preds'],                        # each [2B,1,H,W]: disp_t then disp_t+1
            disp_t=so['final'][:B], disp_t1=so['final'][B:],
        )
        if not run_heads:
            return out

        # ---- geometry at 1/4 resolution (fp32) ----
        with torch.autocast('cuda', enabled=False):
            geo, low = self._geometry(fo['low'][:B].float(), so['low'][:B].float(), so['low'][B:].float(),
                                      K.float(), baseline.float())

        feat = torch.cat([fo['feat'], so['feat'][:B]], 1)
        pix, tokens, tok_pos = self.context(feat, geo)
        mask_logits4 = self.mask_head(pix, tokens, tok_pos)            # [B,K,h4,w4]
        masks4 = mask_logits4.softmax(1)
        bg16 = F.avg_pool2d(masks4[:, :1].detach(), 4).flatten(2).transpose(1, 2)
        pose, pose_logvar = self.pose_head(tokens, tok_pos, bg16)

        mask_logits = F.interpolate(mask_logits4, size=(H, W), mode='bilinear', align_corners=False)
        out.update(
            mask_logits=mask_logits, masks=mask_logits.softmax(1), masks4=masks4, mask_logits4=mask_logits4,
            pose=pose, pose_logvar=pose_logvar, low=low,
        )
        return out

    def _geometry(self, flow4, disp4_t, disp4_t1, K, baseline):
        B, _, h4, w4 = flow4.shape
        K4 = scale_intrinsics(K, 0.25)
        K4[:, :2, 2] = (K[:, :2, 2] + 0.5) / 4 - 0.5     # pixel-centre convention of the 1/4 grid
        fx4 = K4[:, 0, 0]
        depth_t = disp_to_depth(disp4_t, fx4, baseline)
        depth_t1 = disp_to_depth(disp4_t1, fx4, baseline)
        X_t = backproject(depth_t, K4)
        p1 = coords_grid(B, h4, w4, flow4.device, flow4.dtype) + flow4
        X_t1_at, inb = sample(backproject(depth_t1, K4), p1)
        scene_flow = (X_t1_at - X_t).clamp(-5, 5)
        rays = X_t[:, :2] / depth_t.clamp(min=1e-3)
        inv_depth = 1.0 / depth_t.clamp(min=0.5)
        geo = torch.cat([flow4 / fx4.view(-1, 1, 1, 1) * 10, inv_depth * 5, rays, scene_flow], 1)
        if self.detach_geometry:
            geo = geo.detach()
        low = dict(flow=flow4, disp_t=disp4_t, disp_t1=disp4_t1, depth_t=depth_t, depth_t1=depth_t1,
                   K=K4, X_t=X_t, X_t1_at=X_t1_at, inb=inb)
        return geo, low


def pad_inputs(imgs, K, factor=32):
    """Pad images (right/bottom, replicate) to a multiple of `factor`. K is unchanged."""
    H, W = imgs[0].shape[-2:]
    ph, pw = (-H) % factor, (-W) % factor
    if ph or pw:
        imgs = [F.pad(x, (0, pw, 0, ph), mode='replicate') for x in imgs]
    return imgs, (H, W)


def crop_outputs(out, hw):
    H, W = hw
    res = {}
    for k, v in out.items():
        if torch.is_tensor(v) and v.dim() == 4 and v.shape[-1] >= W and v.shape[-2] >= H and v.shape[-1] % 32 == 0:
            res[k] = v[..., :H, :W]
        else:
            res[k] = v
    return res
