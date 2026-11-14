"""Transformer matching branch (UniMatch scale-2 + regression refinement).

Mirrors `UniMatch.forward` for task in {flow, stereo} so that the official pretrained weights load
unchanged, but additionally returns the 1/4-resolution transformer features and low-res estimates
that the mask / pose heads consume.

Pipeline per pair (img0, img1):
  CNN (1/8, 1/4) -> [scale 1/8] pos-enc + 6-layer self/cross transformer -> global correlation
  softmax -> self-attn propagation -> [scale 1/4] warp img1 features by current estimate ->
  transformer -> local correlation softmax (radius 4) -> local propagation -> N x (local corr
  lookup + ConvGRU regression refinement) -> convex upsampling x4.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .unimatch.attention import SelfAttnPropagation
from .unimatch.backbone import CNNEncoder
from .unimatch.geometry import flow_warp
from .unimatch.matching import (global_correlation_softmax, global_correlation_softmax_stereo,
                                local_correlation_softmax, local_correlation_softmax_stereo,
                                local_correlation_with_flow)
from .unimatch.reg_refine import BasicUpdateBlock
from .unimatch.transformer import FeatureTransformer
from .unimatch.utils import feature_add_position, upsample_flow_with_mask

TASK_DEFAULTS = {
    'flow': dict(attn_type='swin', num_reg_refine=6),
    'stereo': dict(attn_type='self_swin2d_cross_swin1d', num_reg_refine=3),
}


class MatchBranch(nn.Module):
    def __init__(self, task='flow', feature_channels=128, num_transformer_layers=6, ffn_dim_expansion=4,
                 attn_splits=(2, 8), corr_radius=(-1, 4), prop_radius=(-1, 1), upsample_factor=4,
                 num_reg_refine=None, grad_checkpoint=True):
        super().__init__()
        assert task in TASK_DEFAULTS
        self.task = task
        self.grad_checkpoint = grad_checkpoint
        self.feature_channels = feature_channels
        self.attn_splits, self.corr_radius, self.prop_radius = attn_splits, corr_radius, prop_radius
        self.num_scales = len(attn_splits)
        self.upsample_factor = upsample_factor
        self.attn_type = TASK_DEFAULTS[task]['attn_type']
        self.num_reg_refine = num_reg_refine or TASK_DEFAULTS[task]['num_reg_refine']

        # attribute names match the UniMatch state dict
        self.backbone = CNNEncoder(output_dim=feature_channels, num_output_scales=self.num_scales)
        self.transformer = FeatureTransformer(num_layers=num_transformer_layers, d_model=feature_channels,
                                              nhead=1, ffn_dim_expansion=ffn_dim_expansion)
        self.feature_flow_attn = SelfAttnPropagation(in_channels=feature_channels)
        self.refine_proj = nn.Conv2d(128, 256, 1)
        self.refine = BasicUpdateBlock(corr_channels=(2 * 4 + 1) ** 2, downsample_factor=upsample_factor,
                                       flow_dim=2 if task == 'flow' else 1, bilinear_up=False)

    def _maybe_ckpt(self, fn, *args, **kw):
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(fn, *args, use_reentrant=False, **kw)
        return fn(*args, **kw)

    def extract_feature(self, img0, img1):
        feats = self._maybe_ckpt(self.backbone, torch.cat((img0, img1), 0))[::-1]  # low -> high resolution
        f0, f1 = zip(*[f.chunk(2, 0) for f in feats])
        return list(f0), list(f1)

    @staticmethod
    def _disp_to_displace(disp):
        return torch.cat((-disp, torch.zeros_like(disp)), 1)

    def forward(self, img0, img1, bidir=False, num_reg_refine=None):
        """img0/img1: ImageNet-normalised [B,3,H,W], H and W divisible by 32.

        Returns dict:
          preds:    list of full-res estimates for sequence supervision (flow [B',2,H,W] /
                    disparity [B',1,H,W]); B' = 2B with bidir (forward then backward flow)
          final:    last full-res estimate
          low:      final estimate at 1/4 resolution (in 1/4-res pixel units)
          feat:     transformer feature of img0 at 1/4 resolution [B,128,H/4,W/4]
          feat1:    transformer feature of img1 at 1/4 resolution
          net:      last ConvGRU hidden state [B',128,H/4,W/4]
        """
        task = self.task
        assert not (bidir and task != 'flow')
        num_reg_refine = num_reg_refine or self.num_reg_refine
        preds = []
        f0_list, f1_list = self.extract_feature(img0, img1)
        flow = None

        for s in range(self.num_scales):
            feature0, feature1 = f0_list[s], f1_list[s]
            if bidir and s > 0:
                feature0, feature1 = torch.cat((feature0, feature1), 0), torch.cat((feature1, feature0), 0)
            feature0_ori, feature1_ori = feature0, feature1
            up = self.upsample_factor * (2 ** (self.num_scales - 1 - s))

            if s > 0:
                flow = F.interpolate(flow, scale_factor=2, mode='bilinear', align_corners=True) * 2
            if flow is not None:
                flow = flow.detach()
                displace = self._disp_to_displace(flow) if task == 'stereo' else flow
                feature1 = flow_warp(feature1, displace)

            feature0, feature1 = feature_add_position(feature0, feature1, self.attn_splits[s], self.feature_channels)
            feature0, feature1 = self._maybe_ckpt(self.transformer, feature0, feature1, attn_type=self.attn_type,
                                                  attn_num_splits=self.attn_splits[s])

            r = self.corr_radius[s]
            if task == 'flow':
                pred = (global_correlation_softmax(feature0, feature1, bidir)[0] if r == -1
                        else local_correlation_softmax(feature0, feature1, r)[0])
            else:
                pred = (global_correlation_softmax_stereo(feature0, feature1)[0] if r == -1
                        else local_correlation_softmax_stereo(feature0, feature1, r)[0])
            flow = flow + pred if flow is not None else pred
            if task == 'stereo':
                flow = flow.clamp(min=0)
            if self.training:
                preds.append(self._bilinear_up(flow, up))

            if bidir and s == 0:
                feature0 = torch.cat((feature0, feature1), 0)
            flow = self.feature_flow_attn(feature0, flow.detach(), local_window_attn=self.prop_radius[s] > 0,
                                          local_window_radius=self.prop_radius[s])
            if self.training and s < self.num_scales - 1:
                preds.append(self._bilinear_up(flow, up))

        # regression refinement at 1/4 resolution
        if self.training:
            preds.append(self._bilinear_up(flow, self.upsample_factor))
        net = None
        for it in range(num_reg_refine):
            flow = flow.detach()
            net, up_mask, flow = self._maybe_ckpt(self._refine_step, feature0_ori, feature1_ori, feature0, flow)
            if self.training or it == num_reg_refine - 1:
                preds.append(upsample_flow_with_mask(flow, up_mask, upsample_factor=self.upsample_factor))

        b = img0.shape[0]
        return dict(preds=preds, final=preds[-1], low=flow, net=net,
                    feat=feature0[:b], feat1=feature1[:b])

    def _refine_step(self, feature0_ori, feature1_ori, feature0, flow):
        displace = self._disp_to_displace(flow) if self.task == 'stereo' else flow
        corr = local_correlation_with_flow(feature0_ori, feature1_ori, flow=displace, local_radius=4)
        net, inp = torch.chunk(self.refine_proj(feature0), 2, 1)
        net, inp = torch.tanh(net), torch.relu(inp)
        net, up_mask, residual = self.refine(net, inp, corr, flow.clone())
        flow = flow + residual
        if self.task == 'stereo':
            flow = flow.clamp(min=0)
        return net, up_mask, flow

    @staticmethod
    def _bilinear_up(flow, factor):
        return F.interpolate(flow, scale_factor=factor, mode='bilinear', align_corners=True) * factor


def load_unimatch_weights(branch: MatchBranch, path, strict=True):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    sd = ckpt.get('model', ckpt)
    missing, unexpected = branch.load_state_dict(sd, strict=strict)
    return missing, unexpected
