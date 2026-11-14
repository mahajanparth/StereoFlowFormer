"""Run the network on consecutive stereo pairs of a sequence (full resolution, padded to /32)."""
import torch
import torch.nn.functional as F

from ..models.model import StereoFlowFormer


def load_model(ckpt_path, device='cuda', pretrained=None):
    """Load a trained checkpoint, or (ckpt_path=None) the raw UniMatch-initialised model (heads untrained)."""
    model = StereoFlowFormer()
    if ckpt_path:
        ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=False)
    elif pretrained:
        model.load_pretrained(*pretrained)
    return model.to(device).eval()


@torch.no_grad()
def infer_pair(model, s, device='cuda'):
    """s: sample dict from KITTIOdometry.load (numpy). Returns dict of cropped full-res torch outputs on GPU."""
    imgs = [torch.from_numpy(s[k])[None].to(device) for k in ('Lt', 'Rt', 'Lt1', 'Rt1')]
    H, W = imgs[0].shape[-2:]
    ph, pw = (-H) % 32, (-W) % 32
    imgs = [F.pad(x, (0, pw, 0, ph), mode='replicate') for x in imgs]
    K = torch.from_numpy(s['K'])[None].to(device)
    bl = torch.tensor([float(s['baseline'])], device=device)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        out = model(*imgs, K, bl)
    res = {k: out[k][0, ..., :H, :W].float() for k in ('flow_fw', 'flow_bw', 'disp_t', 'disp_t1', 'masks')}
    res['pose'] = out['pose'][0].float()
    res['pose_logvar'] = out['pose_logvar'][0].float()
    return res
