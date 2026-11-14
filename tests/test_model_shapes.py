import os

import pytest
import torch

from dvo.models.model import StereoFlowFormer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, 'checkpoints', 'unimatch')

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')


def _inputs(B=2, H=320, W=960):
    imgs = [torch.rand(B, 3, H, W, device='cuda') for _ in range(4)]
    K = torch.tensor([[720.0, 0, W / 2], [0, 720.0, H / 2], [0, 0, 1]], device='cuda').expand(B, 3, 3).clone()
    return imgs, K, torch.full((B,), 0.54, device='cuda')


def test_forward_backward_shapes_bf16():
    model = StereoFlowFormer().cuda().train()
    model.load_pretrained(os.path.join(CKPT, 'gmflow-scale2-regrefine6-things-776ed612.pth'),
                          os.path.join(CKPT, 'gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth'))
    (Lt, Rt, Lt1, Rt1), K, bl = _inputs()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(Lt, Rt, Lt1, Rt1, K, bl)
    B, H, W = 2, 320, 960
    assert out['flow_fw'].shape == (B, 2, H, W) and out['flow_bw'].shape == (B, 2, H, W)
    assert out['disp_t'].shape == (B, 1, H, W) and out['disp_t1'].shape == (B, 1, H, W)
    assert out['masks'].shape == (B, 8, H, W)
    assert torch.allclose(out['masks'].float().sum(1), torch.ones(B, H, W, device='cuda'), atol=1e-2)
    assert out['pose'].shape == (B, 6) and out['pose_logvar'].shape == (B, 6)
    assert len(out['flow_preds']) == 2 + 1 + 1 + 6 and len(out['disp_preds']) == 2 + 1 + 1 + 3
    loss = sum(p.float().abs().mean() for p in out['flow_preds']) + out['masks'].float().var() + out['pose'].float().sum()
    loss.backward()
    assert model.pose_head.out.weight.grad is not None
    assert model.mask_head.query.grad is not None
    print('peak mem GB', torch.cuda.max_memory_allocated() / 1e9)
