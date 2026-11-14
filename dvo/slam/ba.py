"""Sparse stereo bundle adjustment with scipy.optimize.least_squares.

Variables: keyframe poses T_cw (world -> camera) as (rotvec, t), landmark positions X_w.
Residuals:
  * stereo reprojection (u_l, v, u_r) of each observation, divided by pixel sigma
  * optional relative-pose priors between keyframes (e.g. from the network pose head),
    r = sqrt_info * (log_R(R_meas^T R_ji), t_ji - t_meas) with T_ji = T_j T_i^-1
A robust Huber loss is applied element-wise. Reprojection Jacobians are analytic (SO(3) right
Jacobian for the rotvec parameterisation); prior rows use vectorised central differences.
"""
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation


@dataclass
class StereoCamera:
    fx: float
    fy: float
    cx: float
    cy: float
    baseline: float


@dataclass
class BAProblem:
    poses: np.ndarray                 # [C, 6] (rotvec, t) of T_cw
    points: np.ndarray                # [P, 3] world points
    obs_cam: np.ndarray               # [M] int
    obs_pt: np.ndarray                # [M] int
    obs_uvr: np.ndarray               # [M, 3] (u_left, v, u_right)
    fixed_cams: tuple = (0,)
    prior_ij: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), int))   # [Q, 2]
    prior_T: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))         # [Q, 6] T_ji as (rotvec, t)
    prior_sqrt_info: np.ndarray = field(default_factory=lambda: np.zeros((0, 6))) # [Q, 6]


def project_stereo(cam: StereoCamera, poses, points, obs_cam, obs_pt):
    R = Rotation.from_rotvec(poses[obs_cam, :3])
    Xc = R.apply(points[obs_pt]) + poses[obs_cam, 3:]
    z = np.maximum(Xc[:, 2], 1e-3)
    u = cam.fx * Xc[:, 0] / z + cam.cx
    v = cam.fy * Xc[:, 1] / z + cam.cy
    ur = u - cam.fx * cam.baseline / z
    return np.stack([u, v, ur], 1), Xc[:, 2]


def relative_pose(poses, i, j):
    """T_ji = T_j * T_i^-1 for T_cw poses, returned as rotations and translations."""
    return _relative(poses[i], poses[j])


def _relative(pi, pj):
    Ri = Rotation.from_rotvec(pi[:, :3])
    Rj = Rotation.from_rotvec(pj[:, :3])
    Rji = Rj * Ri.inv()
    tji = pj[:, 3:] - Rji.apply(pi[:, 3:])
    return Rji, tji


def _skew(v):
    o = np.zeros(len(v))
    return np.stack([o, -v[:, 2], v[:, 1], v[:, 2], o, -v[:, 0], -v[:, 1], v[:, 0], o], 1).reshape(-1, 3, 3)


def _so3_right_jacobian(w):
    th = np.linalg.norm(w, axis=1)[:, None, None]
    W = _skew(w)
    small = th < 1e-6
    th = np.where(small, 1.0, th)
    a = np.where(small, 0.5, (1 - np.cos(th)) / th ** 2)
    b = np.where(small, 1 / 6, (th - np.sin(th)) / th ** 3)
    return np.eye(3)[None] - a * W + b * (W @ W)


class StereoBA:
    def __init__(self, cam: StereoCamera, pixel_sigma=1.0, huber_px=2.0):
        self.cam = cam
        self.pixel_sigma = pixel_sigma
        self.huber = huber_px / pixel_sigma

    def _unpack(self, x, prob, free_cams):
        poses = prob.poses.copy()
        nc = len(free_cams)
        poses[free_cams] = x[:6 * nc].reshape(nc, 6)
        points = x[6 * nc:].reshape(-1, 3)
        return poses, points

    @staticmethod
    def _prior_res(pi, pj, prob):
        Rji, tji = _relative(pi, pj)
        Rm = Rotation.from_rotvec(prob.prior_T[:, :3])
        r_rot = (Rm.inv() * Rji).as_rotvec()
        r_t = tji - prob.prior_T[:, 3:]
        return np.concatenate([r_rot, r_t], 1) * prob.prior_sqrt_info  # [Q, 6]

    def _residuals(self, x, prob, free_cams):
        poses, points = self._unpack(x, prob, free_cams)
        pred, _ = project_stereo(self.cam, poses, points, prob.obs_cam, prob.obs_pt)
        r = ((pred - prob.obs_uvr) / self.pixel_sigma).ravel()
        if len(prob.prior_ij):
            i, j = prob.prior_ij[:, 0], prob.prior_ij[:, 1]
            r = np.concatenate([r, self._prior_res(poses[i], poses[j], prob).ravel()])
        return r

    def _blocks(self, x, prob, free_cams):
        """Per-observation Jacobian blocks: J_c [M,3,6] (w.r.t. own camera), J_X [M,3,3]; prior blocks
        J_qi, J_qj [Q,6,6]; plus cam_col (column offset of each camera, -1 if fixed)."""
        poses, points = self._unpack(x, prob, free_cams)
        cam = self.cam
        cam_col = -np.ones(len(prob.poses), int)
        cam_col[free_cams] = np.arange(len(free_cams)) * 6
        M, Q = len(prob.obs_cam), len(prob.prior_ij)
        oc, op = prob.obs_cam, prob.obs_pt
        w = poses[oc, :3]
        R = Rotation.from_rotvec(w).as_matrix()                      # [M,3,3]
        X = points[op]
        Xc = np.einsum('mij,mj->mi', R, X) + poses[oc, 3:]
        x_, y_, z = Xc[:, 0], Xc[:, 1], np.maximum(Xc[:, 2], 1e-3)
        o = np.zeros(M)
        dproj = np.stack([
            np.stack([cam.fx / z, o, -cam.fx * x_ / z ** 2], 1),
            np.stack([o, cam.fy / z, -cam.fy * y_ / z ** 2], 1),
            np.stack([cam.fx / z, o, -cam.fx * x_ / z ** 2 + cam.fx * cam.baseline / z ** 2], 1),
        ], 1) / self.pixel_sigma                                      # [M,3,3]
        # d(R(w) X)/dw = -R [X]x Jr(w)
        J_c = np.concatenate([dproj @ (-R @ _skew(X) @ _so3_right_jacobian(w)), dproj], 2)
        J_X = dproj @ R
        J_qi = np.zeros((Q, 6, 6))
        J_qj = np.zeros((Q, 6, 6))
        if Q:
            i, j = prob.prior_ij[:, 0], prob.prior_ij[:, 1]
            pi, pj = poses[i], poses[j]
            eps = 1e-6
            for k in range(6):
                d = np.zeros_like(pi)
                d[:, k] = eps
                J_qi[:, :, k] = (self._prior_res(pi + d, pj, prob) - self._prior_res(pi - d, pj, prob)) / (2 * eps)
                J_qj[:, :, k] = (self._prior_res(pi, pj + d, prob) - self._prior_res(pi, pj - d, prob)) / (2 * eps)
        return J_c, J_X, J_qi, J_qj, cam_col

    def _jacobian(self, x, prob, free_cams):
        """Full sparse Jacobian (used for testing against finite differences)."""
        J_c, J_X, J_qi, J_qj, cam_col = self._blocks(x, prob, free_cams)
        M, Q, P = len(prob.obs_cam), len(prob.prior_ij), len(prob.points)
        n_cam = 6 * len(free_cams)
        oc, op = prob.obs_cam, prob.obs_pt
        rows, cols, vals = [], [], []
        r_idx = (np.arange(M) * 3)[:, None, None] + np.arange(3)[None, :, None]
        sel = cam_col[oc] >= 0
        rr = np.broadcast_to(r_idx[sel], (sel.sum(), 3, 6))
        cc = np.broadcast_to(cam_col[oc][sel][:, None, None] + np.arange(6)[None, None, :], rr.shape)
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(J_c[sel].ravel())
        rr = np.broadcast_to(r_idx, (M, 3, 3))
        cc = np.broadcast_to(n_cam + op[:, None, None] * 3 + np.arange(3)[None, None, :], rr.shape)
        rows.append(rr.ravel()); cols.append(cc.ravel()); vals.append(J_X.ravel())
        for idx, Jq in ((prob.prior_ij[:, 0] if Q else [], J_qi), (prob.prior_ij[:, 1] if Q else [], J_qj)):
            if not Q:
                break
            ok = cam_col[idx] >= 0
            qr = 3 * M + 6 * np.arange(Q)[ok][:, None, None] + np.arange(6)[None, :, None]
            qr = np.broadcast_to(qr, (ok.sum(), 6, 6))
            qc = np.broadcast_to(cam_col[idx][ok][:, None, None] + np.arange(6)[None, None, :], qr.shape)
            rows.append(qr.ravel()); cols.append(qc.ravel()); vals.append(Jq[ok].ravel())
        shape = (3 * M + 6 * Q, n_cam + 3 * P)
        return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=shape).tocsr()

    def _normal_equations(self, x, r, w, prob, free_cams):
        """Assemble Hcc (dense), Hcp (sparse), Hpp blocks [P,3,3], gc, gp directly from Jacobian blocks."""
        J_c, J_X, J_qi, J_qj, cam_col = self._blocks(x, prob, free_cams)
        M, Q, P = len(prob.obs_cam), len(prob.prior_ij), len(prob.points)
        nc = 6 * len(free_cams)
        oc, op = prob.obs_cam, prob.obs_pt
        wr = w[:3 * M].reshape(M, 3)
        rr = r[:3 * M].reshape(M, 3)
        WJc = J_c * wr[..., None]
        WJX = J_X * wr[..., None]
        Hpp = np.zeros((P, 3, 3))
        np.add.at(Hpp, op, np.einsum('mki,mkj->mij', WJX, J_X))
        gp = np.zeros((P, 3))
        np.add.at(gp, op, np.einsum('mki,mk->mi', WJX, rr))
        Hcc = np.zeros((nc, nc))
        gc = np.zeros(nc)
        sel = cam_col[oc] >= 0
        if nc:
            cblk = np.einsum('mki,mkj->mij', WJc[sel], J_c[sel])           # [Ms,6,6]
            cidx = cam_col[oc][sel] // 6
            Hcc_b = np.zeros((len(free_cams), 6, 6))
            np.add.at(Hcc_b, cidx, cblk)
            gc_b = np.zeros((len(free_cams), 6))
            np.add.at(gc_b, cidx, np.einsum('mki,mk->mi', WJc[sel], rr[sel]))
            for c in range(len(free_cams)):
                Hcc[6 * c:6 * c + 6, 6 * c:6 * c + 6] = Hcc_b[c]
            gc = gc_b.ravel()
            # Hcp blocks (6x3) per observation, duplicates summed by coo -> csr
            cp = np.einsum('mki,mkj->mij', WJc[sel], J_X[sel])              # [Ms,6,3]
            rows = np.broadcast_to(cam_col[oc][sel][:, None, None] + np.arange(6)[None, :, None], cp.shape)
            cols = np.broadcast_to(op[sel][:, None, None] * 3 + np.arange(3)[None, None, :], cp.shape)
            Hcp = coo_matrix((cp.ravel(), (rows.ravel(), cols.ravel())), shape=(nc, 3 * P)).tocsr()
            # priors only touch cameras
            if Q:
                wq = w[3 * M:].reshape(Q, 6)
                rq = r[3 * M:].reshape(Q, 6)
                blocks = (prob.prior_ij[:, 0], J_qi), (prob.prior_ij[:, 1], J_qj)
                for a_idx, Ja in blocks:
                    for b_idx, Jb in blocks:
                        Hab = np.einsum('qki,qk,qkj->qij', Ja, wq, Jb)
                        for q in range(Q):
                            ca, cb = cam_col[a_idx[q]], cam_col[b_idx[q]]
                            if ca >= 0 and cb >= 0:
                                Hcc[ca:ca + 6, cb:cb + 6] += Hab[q]
                    ga = np.einsum('qki,qk,qk->qi', Ja, wq, rq)
                    for q in range(Q):
                        ca = cam_col[a_idx[q]]
                        if ca >= 0:
                            gc[ca:ca + 6] += ga[q]
        else:
            Hcp = coo_matrix((0, 3 * P)).tocsr()
        return Hcc, Hcp, Hpp, gc, gp

    def _robust(self, r):
        a = np.abs(r)
        k = self.huber
        cost = np.where(a <= k, 0.5 * r ** 2, k * a - 0.5 * k ** 2).sum()
        w = np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))       # IRLS weights
        return cost, w

    def solve(self, prob: BAProblem, max_iters=30, verbose=0, tol=1e-7):
        """Levenberg-Marquardt with Schur complement on landmarks and IRLS Huber weights."""
        C = len(prob.poses)
        free_cams = np.array([c for c in range(C) if c not in set(prob.fixed_cams)], int)
        nc = 6 * len(free_cams)
        P = len(prob.points)
        x = np.concatenate([prob.poses[free_cams].ravel(), prob.points.ravel()])
        r = self._residuals(x, prob, free_cams)
        r0 = r[:3 * len(prob.obs_cam)].copy()
        cost, w = self._robust(r)
        lam = 1e-4
        it = 0
        for it in range(max_iters):
            Hcc, Hcp, Hpp, gc, gp = self._normal_equations(x, r, w, prob, free_cams)
            Hcp_dense = _to_gpu_dense(Hcp) if nc else None
            improved = False
            for _ in range(10):
                Hpp_d = Hpp + lam * np.einsum('pii->pi', Hpp)[:, :, None] * np.eye(3) + 1e-9 * np.eye(3)
                Hpp_inv = np.linalg.inv(Hpp_d)
                if nc:
                    Hcc_d = Hcc + lam * np.diag(np.diag(Hcc)) + 1e-9 * np.eye(nc)
                    S, b = _schur(Hcc_d, Hcp, Hpp_inv, gc, gp, Hcp_dense)
                    dc = np.linalg.solve(S, b)
                    dp = np.einsum('pij,pj->pi', Hpp_inv, -gp - (Hcp.T @ dc).reshape(P, 3))
                else:
                    dc = np.zeros(0)
                    dp = np.einsum('pij,pj->pi', Hpp_inv, -gp)
                dx = np.concatenate([dc, dp.ravel()])
                x_new = x + dx
                r_new = self._residuals(x_new, prob, free_cams)
                cost_new, w_new = self._robust(r_new)
                if cost_new < cost:
                    rel = (cost - cost_new) / max(cost, 1e-12)
                    x, r, cost, w = x_new, r_new, cost_new, w_new
                    lam = max(lam / 3, 1e-9)
                    improved = True
                    break
                lam *= 4
            if verbose:
                print(f'BA it {it}: cost {cost:.4f} lam {lam:.1e}')
            if not improved or rel < tol or np.abs(dx).max() < 1e-10:
                break
        poses, points = self._unpack(x, prob, free_cams)
        r1 = r[:3 * len(prob.obs_cam)]
        info = dict(cost0=float(np.median(np.abs(r0))), cost1=float(np.median(np.abs(r1))), iters=it + 1)
        return poses, points, info


def _to_gpu_dense(Hcp, max_bytes=4e9):
    """Dense float64 copy of Hcp on the GPU when it fits, else None (sparse CPU fallback)."""
    if not torch.cuda.is_available() or Hcp.shape[0] * Hcp.shape[1] * 8 > max_bytes:
        return None
    coo = Hcp.tocoo()
    D = torch.zeros(Hcp.shape, dtype=torch.float64, device='cuda')
    D[torch.from_numpy(coo.row).cuda(), torch.from_numpy(coo.col).cuda()] = torch.from_numpy(coo.data).cuda()
    return D


def _schur(Hcc_d, Hcp, Hpp_inv, gc, gp, Hcp_dense=None):
    """Reduced camera system S dc = b with S = Hcc - Hcp Hpp^-1 Hcp^T, b = -gc + Hcp Hpp^-1 gp."""
    P = len(Hpp_inv)
    if Hcp_dense is not None:
        nc = Hcp_dense.shape[0]
        Hinv = torch.from_numpy(Hpp_inv).cuda()
        HcpE = torch.einsum('cpi,pij->cpj', Hcp_dense.view(nc, P, 3), Hinv).reshape(nc, 3 * P)
        S = torch.from_numpy(Hcc_d).cuda() - HcpE @ Hcp_dense.T
        b = -torch.from_numpy(gc).cuda() + HcpE @ torch.from_numpy(gp.ravel()).cuda()
        return S.cpu().numpy(), b.cpu().numpy()
    E = _block_diag_sparse(Hpp_inv)
    HcpE = (Hcp @ E).tocsr()
    return Hcc_d - (HcpE @ Hcp.T).toarray(), -gc + HcpE @ gp.ravel()


def _block_diag_sparse(blocks):
    """[P,3,3] dense blocks -> sparse block-diagonal matrix."""
    P = len(blocks)
    base = np.arange(P)[:, None, None] * 3
    rows = np.broadcast_to(base + np.arange(3)[None, :, None], (P, 3, 3))
    cols = np.broadcast_to(base + np.arange(3)[None, None, :], (P, 3, 3))
    return coo_matrix((blocks.ravel(), (rows.ravel(), cols.ravel())), shape=(3 * P, 3 * P)).tocsr()
