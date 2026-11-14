"""Stereo visual odometry back-end driven by the network outputs.

Per frame pair (t, t+1) the network gives flow fw/bw, disparity for t and t+1, motion masks and the
pose-head relative motion T_{t->t+1} (+ log-variance). The back-end:
  1. propagates tracked 2D points of active landmarks with the forward flow; drops points failing the
     forward-backward check, leaving the image, or landing on dynamic (non-background) pixels;
  2. estimates T_cw(t+1) with PnP-RANSAC on landmark 3D <-> tracked 2D, initialised from the pose-head
     prediction (falls back to the pose head when inliers are scarce) - or uses the pose head only;
  3. creates a keyframe when tracks thin out / parallax grows: records stereo observations
     (u, v, u - disp) of tracked landmarks and spawns new landmarks from the disparity map;
  4. runs local BA over the last `window` keyframes (+ pose-head relative-pose priors) and a global BA
     at the end; non-keyframe poses are re-attached to their reference keyframe.
Poses are T_cw (world -> camera) 4x4; the world frame is the first left camera.
"""
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation

from ..geometry.dense_pnp import dense_pnp
from ..geometry.warp import fb_consistency, flow_warp
from .ba import BAProblem, StereoBA, StereoCamera


@dataclass
class VOConfig:
    mode: str = 'pnp_ba_prior'          # pose_head | dense | pnp | pnp_ba | pnp_ba_prior
    init: str = 'dense'                 # frame-to-frame motion: dense (robust dense PnP on flow+disp) | pose_head
    prior_source: str = 'pose_head'     # relative-pose priors in BA: pose_head (learned variance) | dense
    n_features: int = 1500
    min_disp: float = 2.0               # px  (~ max depth = fx*B/min_disp)
    max_depth: float = 40.0
    fb_thresh: float = 1.0              # px forward-backward error
    bg_thresh: float = 0.5              # background-slot probability
    min_inliers: int = 40
    pnp_reproj: float = 2.0
    kf_min_tracks: float = 0.5          # new keyframe when tracked fraction drops below
    kf_max_flow: float = 60.0           # ... or median parallax (px) since last keyframe exceeds
    kf_max_gap: int = 10
    window: int = 8
    ba_iters: int = 15
    global_ba_iters: int = 30
    prior_weight: float = 1.0
    use_masks: bool = True
    dense_stride: int = 4
    dense_var: tuple = (1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4)   # assumed variance of dense estimates (rad^2, m^2)


def pose_to_T(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    return T


def T_to_vec(T):
    return np.concatenate([Rotation.from_matrix(T[:3, :3]).as_rotvec(), T[:3, 3]])


def vec_to_T(v):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(v[:3]).as_matrix()
    T[:3, 3] = v[3:]
    return T


def sample_at(img, pts):
    """img [C,H,W] torch, pts [N,2] torch (x,y) -> [N,C] bilinear samples."""
    H, W = img.shape[-2:]
    g = torch.stack([2 * pts[:, 0] / (W - 1) - 1, 2 * pts[:, 1] / (H - 1) - 1], -1)[None, None]
    return F.grid_sample(img[None], g, align_corners=True)[0, :, 0].T


class StereoVO:
    def __init__(self, K, baseline, img_hw, cfg: VOConfig = VOConfig(), device='cuda'):
        self.cfg, self.device = cfg, device
        self.K = np.asarray(K, np.float64)
        self.baseline = float(baseline)
        self.H, self.W = img_hw
        self.cam = StereoCamera(self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], self.baseline)
        self.ba = StereoBA(self.cam, pixel_sigma=1.0, huber_px=2.0)
        self.poses = [np.eye(4)]                 # T_cw per frame
        self.rel_net = []                        # pose-head T_{t->t+1}
        self.rel_var = []                        # per-dim variance (rad^2, m^2)
        self.kf_frames = []                      # frame index of each keyframe
        self.kf_ref = [0]                        # reference keyframe index per frame
        self.lm_X = np.zeros((0, 3))             # landmark world positions
        self.obs = []                            # (kf_idx, lm_id, u, v, ur)
        self.trk_id = np.zeros(0, int)           # active tracks
        self.trk_uv = torch.zeros(0, 2, device=device)
        self.kf_trk_count = 0
        self.kf_uv0 = torch.zeros(0, 2, device=device)
        self.stats = []

    # ---------------------------------------------------------------- helpers
    def _backproject(self, uv, disp):
        z = self.K[0, 0] * self.baseline / disp
        x = (uv[:, 0] - self.K[0, 2]) * z / self.K[0, 0]
        y = (uv[:, 1] - self.K[1, 2]) * z / self.K[1, 1]
        return np.stack([x, y, z], 1)

    def _static(self, net, key='masks'):
        return net[key][:1] if self.cfg.use_masks else None

    def _add_keyframe(self, frame, disp, bg):
        """Register keyframe: observations of current tracks + new landmarks in empty regions."""
        kf = len(self.kf_frames)
        self.kf_frames.append(frame)
        T_cw = self.poses[frame]
        # observations of existing tracks
        if len(self.trk_id):
            d = sample_at(disp, self.trk_uv)[:, 0].cpu().numpy()
            uv = self.trk_uv.cpu().numpy()
            ok = d > self.cfg.min_disp
            for lid, (u, v), dd in zip(self.trk_id[ok], uv[ok], d[ok]):
                self.obs.append((kf, lid, u, v, u - dd))
            self.trk_id, self.trk_uv = self.trk_id[ok], self.trk_uv[torch.from_numpy(ok).to(self.device)]
        # spawn new landmarks on a grid, avoiding cells that already hold tracks
        need = self.cfg.n_features - len(self.trk_id)
        if need > 50:
            cell = int(np.sqrt(self.H * self.W / self.cfg.n_features))
            occupied = set()
            for u, v in self.trk_uv.cpu().numpy():
                occupied.add((int(v) // cell, int(u) // cell))
            ys, xs = np.mgrid[cell // 2:self.H - 4:cell, cell // 2:self.W - 4:cell]
            cand = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float64)
            free = np.array([(int(v) // cell, int(u) // cell) not in occupied for u, v in cand])
            cand = cand[free]
            # jitter to the strongest gradient inside the cell would be nicer; grid suffices for dense flow
            ct = torch.from_numpy(cand).float().to(self.device)
            d = sample_at(disp, ct)[:, 0].cpu().numpy()
            ok = (d > max(self.cfg.min_disp, self.K[0, 0] * self.baseline / self.cfg.max_depth))
            if bg is not None:
                ok &= sample_at(bg, ct)[:, 0].cpu().numpy() > self.cfg.bg_thresh
            cand, d = cand[ok], d[ok]
            Xc = self._backproject(cand, d)
            T_wc = np.linalg.inv(T_cw)
            Xw = Xc @ T_wc[:3, :3].T + T_wc[:3, 3]
            ids = np.arange(len(self.lm_X), len(self.lm_X) + len(Xw))
            self.lm_X = np.concatenate([self.lm_X, Xw])
            for lid, (u, v), dd in zip(ids, cand, d):
                self.obs.append((kf, lid, u, v, u - dd))
            self.trk_id = np.concatenate([self.trk_id, ids])
            self.trk_uv = torch.cat([self.trk_uv, torch.from_numpy(cand).float().to(self.device)])
        self.kf_trk_count = len(self.trk_id)
        self.kf_uv0 = self.trk_uv.clone()
        return kf

    def _dense_rel(self, net):
        """Robust dense PnP T_{t->t+1} from flow + disparity of frame t on a strided grid (static, fb-consistent)."""
        cfg, st = self.cfg, self.cfg.dense_stride
        disp = net['disp_t'][0, ::st, ::st]
        h, w = disp.shape
        ys, xs = torch.meshgrid(torch.arange(h, device=disp.device) * st, torch.arange(w, device=disp.device) * st,
                                indexing='ij')
        flow = net['flow_fw'][:, ::st, ::st]
        ok = fb_consistency(net['flow_fw'][None], net['flow_bw'][None])[0, 0, ::st, ::st] > 0.5
        z = self.K[0, 0] * self.baseline / disp.clamp(min=1e-3)
        ok &= (disp > cfg.min_disp) & (z < 1.5 * cfg.max_depth) & (xs - disp >= 0)
        qx, qy = xs + flow[0], ys + flow[1]
        ok &= (qx >= 0) & (qx <= self.W - 1) & (qy >= 0) & (qy <= self.H - 1)
        if cfg.use_masks:
            ok &= net['masks'][0, ::st, ::st] > cfg.bg_thresh
        X = torch.stack([(xs - self.K[0, 2]) * z / self.K[0, 0], (ys - self.K[1, 2]) * z / self.K[1, 1], z], 0)
        Kt = torch.from_numpy(self.K).to(disp)[None]
        R, t, _ = dense_pnp(X.reshape(1, 3, -1), torch.stack([qx, qy], 0).reshape(1, 2, -1), Kt,
                            ok.reshape(1, -1).to(disp.dtype), iters=12, sigma=1.0)
        T = np.eye(4)
        T[:3, :3] = R[0].cpu().numpy()
        T[:3, 3] = t[0].cpu().numpy()
        return T

    # ---------------------------------------------------------------- main step
    def start(self, net0):
        """net0: network outputs for pair (0, 1); uses disp_t / masks of frame 0."""
        self._add_keyframe(0, net0['disp_t'], self._static(net0))

    def step(self, net):
        """net: network outputs for pair (t, t+1). Estimates the pose of frame t+1."""
        cfg = self.cfg
        t = len(self.poses) - 1
        rel = vec_to_T(net['pose'].cpu().numpy().astype(np.float64))
        var = np.exp(net['pose_logvar'].cpu().numpy().astype(np.float64))
        var[:3] /= 20.0 ** 2                                # rotation residual was scaled by 20 in training
        rel_dense = self._dense_rel(net) if (cfg.init == 'dense' or cfg.mode == 'dense'
                                             or cfg.prior_source == 'dense') else None
        if cfg.prior_source == 'dense':
            rel, var = rel_dense, np.array(cfg.dense_var)
        self.rel_net.append(rel)
        self.rel_var.append(var)
        use_dense = cfg.mode == 'dense' or (cfg.init == 'dense' and cfg.mode != 'pose_head')
        T_pred = (rel_dense if use_dense else rel) @ self.poses[t]

        # --- track propagation with forward flow + fb-check + static mask
        if len(self.trk_id):
            f = sample_at(net['flow_fw'], self.trk_uv)
            uv1 = self.trk_uv + f
            fb = sample_at(net['flow_bw'], uv1)
            ok = (f + fb).norm(dim=1) < cfg.fb_thresh
            ok &= (uv1[:, 0] >= 0) & (uv1[:, 0] <= self.W - 1) & (uv1[:, 1] >= 0) & (uv1[:, 1] <= self.H - 1)
            if cfg.use_masks:
                ok &= sample_at(net['masks'][:1], self.trk_uv)[:, 0] > cfg.bg_thresh
            okn = ok.cpu().numpy()
            self.trk_id, self.trk_uv = self.trk_id[okn], uv1[ok]
            self.kf_uv0 = self.kf_uv0[ok]

        # --- pose estimate
        n_inl = 0
        T_new = T_pred
        if cfg.mode not in ('pose_head', 'dense') and len(self.trk_id) >= 6:
            obj = self.lm_X[self.trk_id]
            img = self.trk_uv.cpu().numpy().astype(np.float64)
            rvec0 = cv2.Rodrigues(T_pred[:3, :3])[0]
            tvec0 = T_pred[:3, 3:].copy()
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, self.K, None, rvec0, tvec0, useExtrinsicGuess=True,
                                                     iterationsCount=200, reprojectionError=cfg.pnp_reproj,
                                                     confidence=0.999, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok and inl is not None and len(inl) >= cfg.min_inliers:
                inl = inl[:, 0]
                rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], self.K, None, rvec, tvec)
                T_new = pose_to_T(rvec, tvec)
                n_inl = len(inl)
                # drop only gross outliers (landmarks from single-view stereo drift as the camera moves)
                proj = (self.K @ (T_new[:3, :3] @ obj.T + T_new[:3, 3:])).T
                err = np.linalg.norm(proj[:, :2] / np.maximum(proj[:, 2:], 1e-3) - img, axis=1)
                keep = err < 4 * cfg.pnp_reproj
                kt = torch.from_numpy(keep).to(self.device)
                self.trk_id, self.trk_uv, self.kf_uv0 = self.trk_id[keep], self.trk_uv[kt], self.kf_uv0[kt]
        self.poses.append(T_new)
        self.kf_ref.append(len(self.kf_frames) - 1)

        # --- keyframe decision (on frame t+1, whose disparity is net['disp_t1'])
        frac = len(self.trk_id) / max(self.kf_trk_count, 1)
        parallax = (self.trk_uv - self.kf_uv0).norm(dim=1).median().item() if len(self.trk_id) else 1e9
        gap = (t + 1) - self.kf_frames[-1]
        is_kf = frac < cfg.kf_min_tracks or parallax > cfg.kf_max_flow or gap >= cfg.kf_max_gap
        if is_kf and cfg.mode not in ('pose_head', 'dense'):
            bg1 = None
            if cfg.use_masks:  # background mask of frame t+1: warp frame-t mask with the backward flow
                bg1, _ = flow_warp(net['masks'][None, :1], net['flow_bw'][None])
                bg1 = bg1[0]
            self._add_keyframe(t + 1, net['disp_t1'], bg1)
            self.kf_ref[-1] = len(self.kf_frames) - 1
            if cfg.mode in ('pnp_ba', 'pnp_ba_prior') and len(self.kf_frames) >= 3:
                self.bundle_adjust(self.cfg.window, self.cfg.ba_iters)
        self.stats.append(dict(frame=t + 1, tracks=len(self.trk_id), inliers=n_inl, kf=is_kf))
        return T_new

    # ---------------------------------------------------------------- BA
    def _kf_prior(self, a, b):
        """Pose-head relative motion from keyframe a to b (chained) and its per-dim variance."""
        fa, fb = self.kf_frames[a], self.kf_frames[b]
        T = np.eye(4)
        var = np.zeros(6)
        for k in range(fa, fb):
            T = self.rel_net[k] @ T
            var += self.rel_var[k]
        return T_to_vec(T), var

    def bundle_adjust(self, window=None, iters=15):
        n_kf = len(self.kf_frames)
        kfs = list(range(n_kf)) if window is None else list(range(max(0, n_kf - window), n_kf))
        kf_set = {k: i for i, k in enumerate(kfs)}
        obs = [o for o in self.obs if o[0] in kf_set]
        if len(obs) < 20:
            return None
        lm_ids = np.unique([o[1] for o in obs])
        lm_map = {l: i for i, l in enumerate(lm_ids)}
        old_T = [self.poses[self.kf_frames[k]] for k in kfs]
        prob = BAProblem(
            poses=np.array([T_to_vec(T) for T in old_T]),
            points=self.lm_X[lm_ids].copy(),
            obs_cam=np.array([kf_set[o[0]] for o in obs]),
            obs_pt=np.array([lm_map[o[1]] for o in obs]),
            obs_uvr=np.array([o[2:] for o in obs], np.float64),
            fixed_cams=(0,) if window is None or kfs[0] == 0 else (0, 1) if len(kfs) > 3 else (0,),
        )
        if self.cfg.mode == 'pnp_ba_prior' and len(kfs) > 1:
            ij, meas, sq = [], [], []
            for a, b in zip(kfs[:-1], kfs[1:]):
                v, var = self._kf_prior(a, b)
                ij.append((kf_set[a], kf_set[b]))
                meas.append(v)
                sq.append(self.cfg.prior_weight / np.sqrt(var + 1e-8))
            prob.prior_ij, prob.prior_T, prob.prior_sqrt_info = np.array(ij), np.array(meas), np.array(sq)
        poses, points, info = self.ba.solve(prob, max_iters=iters)
        self.lm_X[lm_ids] = points
        # update keyframe poses and re-attach non-keyframes (keep their pose relative to the ref keyframe)
        new_T = [vec_to_T(p) for p in poses]
        for f in range(self.kf_frames[kfs[0]], len(self.poses)):
            ref = self.kf_ref[f]
            if ref in kf_set:
                self.poses[f] = corr_apply(self.poses[f], old_T[kf_set[ref]], new_T[kf_set[ref]])
        return info

    def finish(self):
        if self.cfg.mode in ('pnp_ba', 'pnp_ba_prior') and len(self.kf_frames) >= 3:
            return self.bundle_adjust(None, self.cfg.global_ba_iters)

    def trajectory_wc(self):
        """Camera-to-world poses [N,4,4]."""
        return np.array([np.linalg.inv(T) for T in self.poses])


def corr_apply(T_f, T_ref_old, T_ref_new):
    """Keep the relative pose of frame f w.r.t. its reference keyframe: T_f' = (T_f T_ref_old^-1) T_ref_new."""
    return T_f @ np.linalg.inv(T_ref_old) @ T_ref_new
