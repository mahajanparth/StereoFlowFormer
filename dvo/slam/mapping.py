"""Fuse keyframe RGB-D (depth from stereo disparity) into a global map with the optimised poses."""
import numpy as np
import open3d as o3d


def keyframe_points(rgb, disp, K, baseline, static=None, max_depth=30.0, min_disp=1.0, stride=2):
    """rgb [3,H,W] in [0,1], disp [H,W] -> camera-frame points [N,3], colours [N,3]."""
    H, W = disp.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = disp[ys, xs]
    z = K[0, 0] * baseline / np.maximum(d, 1e-3)
    ok = (d > min_disp) & (z < max_depth)
    if static is not None:
        ok &= static[ys, xs] > 0.5
    ok[: int(0.25 * ok.shape[0])] = False        # top quarter of KITTI frames is mostly sky / far
    x = (xs[ok] - K[0, 2]) * z[ok] / K[0, 0]
    y = (ys[ok] - K[1, 2]) * z[ok] / K[1, 1]
    return np.stack([x, y, z[ok]], 1), rgb[:, ys[ok], xs[ok]].T


class PointMap:
    def __init__(self, voxel=0.1):
        self.voxel = voxel
        self.pcd = o3d.geometry.PointCloud()
        self._pending = []

    def add(self, pts_c, cols, T_wc):
        pw = pts_c @ T_wc[:3, :3].T + T_wc[:3, 3]
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pw)
        p.colors = o3d.utility.Vector3dVector(np.clip(cols, 0, 1))
        self._pending.append(p)
        if len(self._pending) >= 20:
            self._flush()

    def _flush(self):
        for p in self._pending:
            self.pcd += p
        self._pending = []
        self.pcd = self.pcd.voxel_down_sample(self.voxel)

    def save(self, path):
        self._flush()
        o3d.io.write_point_cloud(path, self.pcd)
        return len(self.pcd.points)


class TSDFMap:
    """Open3D scalable TSDF (better surfaces, use for short sequences)."""

    def __init__(self, voxel=0.1, trunc=0.4, max_depth=30.0):
        self.vol = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel, sdf_trunc=trunc, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        self.max_depth = max_depth

    def add(self, rgb, depth, K, T_cw):
        H, W = depth.shape
        color = o3d.geometry.Image(np.ascontiguousarray((rgb.transpose(1, 2, 0) * 255).astype(np.uint8)))
        dimg = o3d.geometry.Image(np.ascontiguousarray(depth.astype(np.float32)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, dimg, depth_scale=1.0,
                                                                  depth_trunc=self.max_depth,
                                                                  convert_rgb_to_intensity=False)
        intr = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
        self.vol.integrate(rgbd, intr, T_cw)

    def save(self, path):
        mesh = self.vol.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(path, mesh)
        return len(mesh.vertices)
