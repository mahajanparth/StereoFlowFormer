# StereoFlowFormer

A transformer network that takes **two stereo pairs** (left/right at time *t* and *t+1*) and predicts:

- **Optical flow** (L_t → L_t+1, forward and backward) from a global 4D correlation volume over transformer features, refined locally
- **Disparity / inverse depth** for both frames from 1D epipolar correlation
- **Motion-instance masks**, learned without labels: pixels whose flow is not explained by the camera's rigid motion become moving objects
- **Ego-motion** with uncertainty

These outputs feed a stereo VO back-end: flow-based tracking, PnP, sparse stereo bundle adjustment and RGB-D mapping.

![flow and stereo demo](demo.gif)

*KITTI odometry seq. 08. Top: left input. Middle: optical flow. Bottom: disparity.* ([mp4](demo.mp4))

## Current numbers (pretrained baseline)

Flow and stereo branches as initialised from UniMatch, before any fine-tuning.

**KITTI 2015 training set** (200 frames, zero-shot, full resolution)

| Task | EPE | Outliers |
|---|---|---|
| Optical flow | 5.50 px | Fl-all 17.4 % |
| Disparity | 1.22 px | D1-all 5.70 % |

**Virtual KITTI 2 validation** (320×960 crops, last 50 frames of each scene)

| Task | EPE | Outliers |
|---|---|---|
| Optical flow | 0.57 px | Fl 2.8 % |
| Disparity | 0.70 px | D1 3.6 % |

Speed: about 160 ms per forward pass of the full network (bidirectional flow, two stereo pairs and heads) at 320×960 on an RTX 5090 in bf16.

## How it works

```
L_t R_t L_t+1 R_t+1
   │   flow branch (L_t, L_t+1, bidirectional)      stereo branch (L,R) at t and t+1
   │   CNN 1/8,1/4 → transformer → global corr       CNN → transformer (1D cross) → 1D corr
   │   → propagation → local corr + GRU ×6           → propagation → local corr + GRU ×3
   ▼
 flow fw/bw, disp_t, disp_t+1 ─┬─ geometry @1/4 (flow/f, inv-depth, rays, 3D scene flow)
                               ├─ mask head  → masks (slot 0 = background)
                               └─ pose head  → (rotvec, t), log-variance
```

- **Masks (self-supervised):** ego-motion is estimated per batch by dense robust PnP between 3D points of frame *t* (stereo depth) and their flow correspondences. Pixels whose flow disagrees with that rigid motion, beyond an uncertainty-aware threshold, are labelled moving. Connected components become instances. The mask head is trained on these pseudo-labels with Hungarian-matched cross-entropy + Dice.
- **Losses:** sequence L1 on flow and disparity (Virtual KITTI 2), SSIM + L1 photometric losses with forward-backward occlusion masks (KITTI), edge-aware smoothness, Gaussian NLL for the pose head, and rigid-flow consistency between pose head and flow.
- **Back-end:** points tracked with flow (forward-backward and static-mask checks), PnP-RANSAC initialised by dense PnP, local and global stereo bundle adjustment (Levenberg-Marquardt with Schur complement, Huber loss, optional pose-head priors), point-cloud / TSDF fusion.

## Layout
```
dvo/models     match_branch.py (UniMatch-equivalent branch), heads.py, model.py, unimatch/ (vendored, MIT)
dvo/geometry   se3, warp, procrustes (weighted Kabsch), dense_pnp
dvo/losses     supervised, photometric, motion_seg, pose
dvo/data       vkitti2.py, kitti.py (odometry + KITTI 2015)
dvo/slam       frontend.py, vo.py, ba.py, mapping.py
dvo/eval       metrics.py, traj_eval.py
scripts        download_data.py, train.py, eval.py, run_vo.py, visualize.py
configs        stage1a_heads.yaml → vkitti2_sup.yaml (stage 1b) → kitti_selfsup.yaml (stage 2)
```

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install -r requirements.txt
# UniMatch weights (Things / SceneFlow, no KITTI 2015 exposure)
mkdir -p checkpoints/unimatch && cd checkpoints/unimatch && \
  B=https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained && \
  curl -LO $B/gmflow-scale2-regrefine6-things-776ed612.pth && \
  curl -LO $B/gmstereo-scale2-regrefine3-resumeflowthings-sceneflow-f724fee6.pth && cd -
# data (~25 GB on disk: Virtual KITTI 2 clone, KITTI odometry 00-10, KITTI 2015)
.venv/bin/python scripts/download_data.py --root data
```

## Training
```bash
.venv/bin/python scripts/train.py --config configs/stage1a_heads.yaml    # heads only, branches frozen
.venv/bin/python scripts/train.py --config configs/vkitti2_sup.yaml      # stage 1b: joint, Virtual KITTI 2
.venv/bin/python scripts/train.py --config configs/kitti_selfsup.yaml    # stage 2: KITTI self-supervised
```
Override any config key with `--set train.iters=500 data.vkitti2_scenes=[Scene01]`.

## Evaluation, VO and maps
```bash
.venv/bin/python scripts/eval.py --dataset kitti2015 --ckpt runs/stage2_kitti_selfsup/checkpoints/latest.pth
.venv/bin/python scripts/run_vo.py --seq 09 --ckpt ... --modes dense pnp pnp_ba pnp_ba_prior --map
.venv/bin/python scripts/visualize.py --seq 09 --frames 0 200 400 --ckpt ...
.venv/bin/python -m pytest
```

## Status

Work in progress. The demo and numbers above use the flow and stereo branches initialised from
[UniMatch](https://github.com/autonomousvision/unimatch) pretrained weights.
Training of the mask and pose heads on Virtual KITTI 2 and self-supervised KITTI fine-tuning are underway;
numbers will be updated when they finish.

Known limitations: masks separate independently moving objects only (parked cars are background), and the
pretrained stereo branch underestimates far-range disparity, which makes the VO scale about 1% too large.
