"""Our MatchBranch must reproduce the official UniMatch forward pass with the same weights."""
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, 'checkpoints', 'unimatch')
UNIMATCH_REPO = os.path.join(ROOT, 'third_party', 'unimatch')

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and os.path.isdir(UNIMATCH_REPO)),
                                reason='needs CUDA and third_party/unimatch')


def _official():
    sys.path.insert(0, UNIMATCH_REPO)
    from unimatch.unimatch import UniMatch
    return UniMatch


@pytest.mark.parametrize('task,ckpt,bidir', [
    ('flow', 'gmflow-scale2-regrefine6-things-776ed612.pth', False),
    ('flow', 'gmflow-scale2-regrefine6-things-776ed612.pth', True),
    ('stereo', 'gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth', False),
])
def test_matches_official(task, ckpt, bidir):
    from dvo.models.match_branch import MatchBranch, TASK_DEFAULTS, load_unimatch_weights
    UniMatch = _official()
    torch.manual_seed(0)
    path = os.path.join(CKPT, ckpt)
    ref = UniMatch(num_scales=2, upsample_factor=4, reg_refine=True, task=task).cuda().eval()
    ref.load_state_dict(torch.load(path, map_location='cpu', weights_only=False)['model'])
    ours = MatchBranch(task=task).cuda().eval()
    load_unimatch_weights(ours, path)

    img0 = torch.rand(1, 3, 128, 256, device='cuda') * 255
    img1 = torch.roll(img0, 3, -1)
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)
    n0, n1 = (img0 / 255 - mean) / std, (img1 / 255 - mean) / std
    with torch.no_grad():
        kw = dict(attn_type=TASK_DEFAULTS[task]['attn_type'], attn_splits_list=[2, 8], corr_radius_list=[-1, 4],
                  prop_radius_list=[-1, 1], num_reg_refine=TASK_DEFAULTS[task]['num_reg_refine'], task=task,
                  pred_bidir_flow=bidir)
        # official flow model normalises internally from [0,255]; stereo expects normalised input
        out_ref = ref(img0, img1, **kw) if task == 'flow' else ref(n0, n1, **kw)
        out = ours(n0, n1, bidir=bidir)
    a = out_ref['flow_preds'][-1]
    b = out['final'].squeeze(1) if task == 'stereo' else out['final']
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-3), (a - b).abs().max()
