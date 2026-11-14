"""Trajectory metrics: KITTI odometry t_err (%) / r_err (deg/100m) and ATE (SE(3)-aligned RMSE)."""
import numpy as np

LENGTHS = (100, 200, 300, 400, 500, 600, 700, 800)


def _dists(poses):
    d = [0.0]
    for i in range(1, len(poses)):
        d.append(d[-1] + np.linalg.norm(poses[i, :3, 3] - poses[i - 1, :3, 3]))
    return np.array(d)


def _rot_err(T):
    return np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))


def kitti_errors(gt, est, step=10):
    """gt, est: [N,4,4] camera-to-world. Returns mean translational error (%) and rotational (deg/100m)."""
    dist = _dists(gt)
    errs = []
    for first in range(0, len(gt), step):
        for L in LENGTHS:
            last = np.searchsorted(dist, dist[first] + L)
            if last >= len(gt):
                continue
            dg = np.linalg.inv(gt[first]) @ gt[last]
            de = np.linalg.inv(est[first]) @ est[last]
            e = np.linalg.inv(de) @ dg
            errs.append((np.linalg.norm(e[:3, 3]) / L, _rot_err(e) / L))
    if not errs:
        return dict(t_err=float('nan'), r_err=float('nan'))
    errs = np.array(errs)
    return dict(t_err=float(errs[:, 0].mean() * 100), r_err=float(np.degrees(errs[:, 1].mean()) * 100))


def umeyama(src, dst, with_scale=False):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    S, D = src - mu_s, dst - mu_d
    U, sig, Vt = np.linalg.svd(D.T @ S / len(src))
    E = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        E[2, 2] = -1
    R = U @ E @ Vt
    s = (sig * np.diag(E)).sum() / S.var(0).sum() if with_scale else 1.0
    t = mu_d - s * R @ mu_s
    return R, t, s


def ate(gt, est, with_scale=False):
    R, t, s = umeyama(est[:, :3, 3], gt[:, :3, 3], with_scale)
    aligned = s * est[:, :3, 3] @ R.T + t
    return float(np.sqrt(((aligned - gt[:, :3, 3]) ** 2).sum(1).mean()))


def evaluate(gt, est):
    n = min(len(gt), len(est))
    gt, est = gt[:n], est[:n]
    res = kitti_errors(gt, est)
    res['ate'] = ate(gt, est)
    res['length_m'] = float(_dists(gt)[-1])
    return res


def save_kitti(path, poses_wc):
    np.savetxt(path, poses_wc[:, :3, :].reshape(len(poses_wc), 12), fmt='%.6e')
