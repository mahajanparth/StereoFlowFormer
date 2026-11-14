import numpy as np
from scipy.spatial.transform import Rotation

from dvo.slam.ba import BAProblem, StereoBA, StereoCamera, project_stereo, relative_pose


def make_scene(n_cams=8, n_pts=400, seed=0):
    rng = np.random.default_rng(seed)
    cam = StereoCamera(fx=720.0, fy=720.0, cx=620.0, cy=187.0, baseline=0.54)
    # camera driving forward along +z with slight yaw; poses are T_cw
    poses = []
    for i in range(n_cams):
        R_wc = Rotation.from_euler('y', 0.02 * i)
        c = np.array([0.1 * np.sin(i * 0.3), 0.0, 1.0 * i])
        R_cw = R_wc.inv()
        poses.append(np.concatenate([R_cw.as_rotvec(), -R_cw.apply(c)]))
    poses = np.array(poses)
    pts = np.stack([rng.uniform(-15, 15, n_pts), rng.uniform(-3, 2, n_pts), rng.uniform(8, 45, n_pts)], 1)
    obs_cam, obs_pt = np.meshgrid(np.arange(n_cams), np.arange(n_pts), indexing='ij')
    obs_cam, obs_pt = obs_cam.ravel(), obs_pt.ravel()
    uvr, z = project_stereo(cam, poses, pts, obs_cam, obs_pt)
    keep = (z > 1) & (uvr[:, 0] > 0) & (uvr[:, 0] < 1240) & (uvr[:, 1] > 0) & (uvr[:, 1] < 375)
    uvr = uvr + rng.normal(0, 0.5, uvr.shape)
    return cam, poses, pts, obs_cam[keep], obs_pt[keep], uvr[keep], rng


def rot_err_deg(p, q):
    return np.degrees(np.linalg.norm((Rotation.from_rotvec(p[:, :3]).inv() * Rotation.from_rotvec(q[:, :3])).as_rotvec(), axis=1))


def test_ba_recovers_poses_from_noisy_init():
    cam, poses, pts, oc, op, uvr, rng = make_scene()
    noisy = poses.copy()
    noisy[1:, :3] += rng.normal(0, 0.01, (len(poses) - 1, 3))
    noisy[1:, 3:] += rng.normal(0, 0.15, (len(poses) - 1, 3))
    noisy_pts = pts + rng.normal(0, 0.3, pts.shape)
    # gross outliers
    uvr[:20] += 40
    prob = BAProblem(poses=noisy, points=noisy_pts, obs_cam=oc, obs_pt=op, obs_uvr=uvr)
    est, _, info = StereoBA(cam).solve(prob, max_iters=50)
    assert info['cost1'] < info['cost0']
    assert rot_err_deg(est, poses).max() < 0.1
    assert np.abs(est[:, 3:] - poses[:, 3:]).max() < 0.05


def test_ba_with_relative_pose_priors():
    cam, poses, pts, oc, op, uvr, rng = make_scene(n_cams=6, n_pts=60, seed=1)
    ij = np.array([[k, k + 1] for k in range(len(poses) - 1)])
    meas = []
    for i, j in ij:
        Rji, tji = relative_pose(poses, np.array([i]), np.array([j]))
        meas.append(np.concatenate([Rji.as_rotvec()[0], tji[0]]))
    noisy = poses.copy()
    noisy[1:, 3:] += rng.normal(0, 0.3, (len(poses) - 1, 3))
    prob = BAProblem(poses=noisy, points=pts + rng.normal(0, 0.3, pts.shape), obs_cam=oc, obs_pt=op,
                     obs_uvr=uvr, prior_ij=ij, prior_T=np.array(meas),
                     prior_sqrt_info=np.full((len(ij), 6), 10.0))
    est, _, _ = StereoBA(cam).solve(prob, max_iters=50)
    assert np.abs(est[:, 3:] - poses[:, 3:]).max() < 0.05


def test_analytic_jacobian_matches_finite_difference():
    from scipy.optimize._numdiff import approx_derivative
    cam, poses, pts, oc, op, uvr, rng = make_scene(n_cams=4, n_pts=30, seed=2)
    ij = np.array([[0, 1], [1, 2], [2, 3]])
    prob = BAProblem(poses=poses + rng.normal(0, 0.05, poses.shape), points=pts, obs_cam=oc, obs_pt=op,
                     obs_uvr=uvr, prior_ij=ij, prior_T=rng.normal(0, 0.1, (3, 6)),
                     prior_sqrt_info=np.full((3, 6), 3.0))
    ba = StereoBA(cam)
    free = np.arange(1, 4)
    x = np.concatenate([prob.poses[free].ravel(), prob.points.ravel()])
    J = ba._jacobian(x, prob, free).toarray()
    J_num = approx_derivative(ba._residuals, x, method='3-point', args=(prob, free))
    assert np.abs(J - J_num).max() < 1e-3 * max(1.0, np.abs(J_num).max())
