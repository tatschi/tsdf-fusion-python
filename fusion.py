# Copyright (c) 2018 Andy Zeng

import numpy as np

from numba import njit, prange
from skimage import measure

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    from pycuda.compiler import SourceModule

    FUSION_GPU_MODE = 1
except Exception as err:
    print('Warning: {}'.format(err))
    print('Failed to import PyCUDA. Running fusion in CPU mode.')
    FUSION_GPU_MODE = 0


class TSDFVolume:
    """Volumetric TSDF Fusion of depth Images.
  """

    def __init__(self, vol_bnds, voxel_size, use_gpu=True):
        """Constructor.

    Args:
      vol_bnds (ndarray): An ndarray of shape (3, 2). Specifies the
        xyz bounds (min/max) in meters.
      voxel_size (float): The volume discretization in meters.
    """
        self.gpu_grid = None
        self._n_gpu_loops = None
        self._max_gpu_grid_dim = None
        self._max_gpu_threads_per_block = None
        vol_bnds = np.asarray(vol_bnds)
        assert vol_bnds.shape == (3, 2), "[!] `vol_bnds` should be of shape (3, 2)."

        # Define voxel volume parameters
        self._vol_bnds = vol_bnds
        self._voxel_size = float(voxel_size)
        self._trunc_margin = 5 * self._voxel_size  # truncation on SDF

        # Adjust volume bounds and ensure C-order contiguous
        self._vol_dim = np.ceil((self._vol_bnds[:, 1] - self._vol_bnds[:, 0]) / self._voxel_size).copy(
            order='C').astype(int)
        self._vol_bnds[:, 1] = self._vol_bnds[:, 0] + self._vol_dim * self._voxel_size
        self._vol_origin = self._vol_bnds[:, 0].copy(order='C').astype(np.float32)

        print("Voxel volume size: {} x {} x {} - # points: {:,}".format(
            self._vol_dim[0], self._vol_dim[1], self._vol_dim[2],
            self._vol_dim[0] * self._vol_dim[1] * self._vol_dim[2])
        )

        # Initialize pointers to voxel volume in CPU memory
        self._tsdf_vol = np.ones(self._vol_dim).astype(np.float32)
        # for computing the cumulative moving average of observations per voxel
        self._weight_vol = np.zeros(self._vol_dim).astype(np.float32)

        self.gpu_mode = use_gpu and FUSION_GPU_MODE

        if self.gpu_mode:

            # Cuda kernel function (C++)
            with open("kernels.cpp") as file:
                self._cuda_src_mod = SourceModule(file.read())
            self._cuda_find_voxels = self._cuda_src_mod.get_function("find_voxels_for_points")
            self._cuda_compute_dist_between_point_and_voxel = self._cuda_src_mod \
                .get_function("compute_dist_between_point_and_voxel")
        else:
            # Get voxel grid coordinates
            xv, yv, zv = np.meshgrid(
                range(self._vol_dim[0]),
                range(self._vol_dim[1]),
                range(self._vol_dim[2]),
                indexing='ij'
            )
            self.vox_coords = np.concatenate([
                xv.reshape(1, -1),
                yv.reshape(1, -1),
                zv.reshape(1, -1)
            ], axis=0).astype(int).T

    @staticmethod
    @njit(parallel=True)
    def vox2world(vol_origin, vox_coords, vox_size):
        """Convert voxel grid coordinates to world coordinates.
    """
        vol_origin = vol_origin.astype(np.float32)
        vox_coords = vox_coords.astype(np.float32)
        cam_pts = np.empty_like(vox_coords, dtype=np.float32)
        for i in prange(vox_coords.shape[0]):
            for j in range(3):
                cam_pts[i, j] = vol_origin[j] + (vox_size * vox_coords[i, j])
        return cam_pts

    @staticmethod
    @njit(parallel=True)
    def cam2pix(cam_pts, intr):
        """Convert camera coordinates to pixel coordinates.
    """
        intr = intr.astype(np.float32)
        fx, fy = intr[0, 0], intr[1, 1]
        cx, cy = intr[0, 2], intr[1, 2]
        pix = np.empty((cam_pts.shape[0], 2), dtype=np.int64)
        for i in prange(cam_pts.shape[0]):
            pix[i, 0] = int(np.round((cam_pts[i, 0] / fx) + cx))
            pix[i, 1] = int(np.round((cam_pts[i, 1] / fy) + cy))
        return pix

    @staticmethod
    @njit(parallel=True)
    def integrate_tsdf(tsdf_vol, dist, w_old, obs_weight):
        """Integrate the TSDF volume.
    """
        tsdf_vol_int = np.empty_like(tsdf_vol, dtype=np.float32)
        w_new = np.empty_like(w_old, dtype=np.float32)
        for i in prange(len(tsdf_vol)):
            w_new[i] = w_old[i] + obs_weight
            tsdf_vol_int[i] = (w_old[i] * tsdf_vol[i] + obs_weight * dist[i]) / w_new[i]
        return tsdf_vol_int, w_new

    def integrate(self, point_cloud, colors):
        """Integrate a depth frame into the TSDF volume.

    Args:
      point_cloud (ndarray): A list of 3D points.
      colors (ndarray): A list of RGB colors.
    """
        if len(colors) == 0:
            colors = np.ones(point_cloud.shape)
        if self.gpu_mode:
            self.integrate_gpu_mode(point_cloud, colors)
        else:
            self.integrate_cpu_mode(point_cloud, colors)

    def integrate_cpu_mode(self, point_cloud, colors):
        world_coords = self.vox2world(self._vol_origin, self.vox_coords, self._voxel_size)

        for point, color in zip(point_cloud, colors):
            depth_diff = np.zeros(world_coords.shape[0])
            voxel_index_xy = np.asarray(np.floor((point[:2] - self._vol_origin[:2]) / self._voxel_size),
                                        dtype=np.int64)
            voxel_index_mask = np.logical_and(self.vox_coords[:, 0] == voxel_index_xy[0],
                                              self.vox_coords[:, 1] == voxel_index_xy[1])
            voxel_indices = np.argwhere(voxel_index_mask)

            depth_diff[voxel_indices] = point[2] - world_coords[voxel_indices, 2]

            # TODO maybe optimize this because it is only updating one voxel at a time
            valid_pts = np.logical_and(depth_diff > 0, depth_diff >= -self._trunc_margin)
            dist = np.minimum(1, depth_diff / self._trunc_margin)
            valid_vox_x = self.vox_coords[valid_pts, 0]
            valid_vox_y = self.vox_coords[valid_pts, 1]
            valid_vox_z = self.vox_coords[valid_pts, 2]
            w_old = self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z]
            tsdf_vals = self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z]
            valid_dist = dist[valid_pts]
            obs_weight = color[1]
            tsdf_vol_new, w_new = self.integrate_tsdf(tsdf_vals, valid_dist, w_old, obs_weight)
            self._weight_vol[valid_vox_x, valid_vox_y, valid_vox_z] = w_new
            self._tsdf_vol[valid_vox_x, valid_vox_y, valid_vox_z] = tsdf_vol_new

    def integrate_gpu_mode(self, point_cloud, colors):
        voxels_x, voxels_y = self.find_relevant_voxels_gpu_mode(point_cloud)
        # repeat all values for all possible z values
        voxels_x = np.repeat(voxels_x, self._vol_dim[2]).astype(np.int32)
        voxels_y = np.repeat(voxels_y, self._vol_dim[2]).astype(np.int32)
        points_z = np.repeat(point_cloud[:, 2], self._vol_dim[2]).astype(np.float32)
        colors_rep = np.repeat(colors, self._vol_dim[2], axis=0)
        z_vals = np.arange(self._vol_dim[2])
        voxels_z = np.tile(z_vals, len(point_cloud)).astype(np.int32)
        dists = np.zeros(len(points_z)).astype(np.float32)
        self.init_gpu_grid(len(points_z))
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_compute_dist_between_point_and_voxel(cuda.InOut(self._vol_origin.astype(np.float32)),
                                                            cuda.InOut(np.asarray([
                                                                gpu_loop_idx,
                                                                self._voxel_size,
                                                                self._trunc_margin
                                                            ], np.float32)),
                                                            cuda.InOut(voxels_z),
                                                            cuda.InOut(points_z),
                                                            cuda.InOut(dists),
                                                            block=(self._max_gpu_threads_per_block, 1, 1),
                                                            grid=self.gpu_grid
                                                            )
        for i in range(len(dists)):
            voxel_index = voxels_x[i], voxels_y[i], voxels_z[i]
            if dists[i] == 0:
                continue
            w_old = self._weight_vol[voxel_index]
            obs_weight = colors_rep[i, 1]
            w_new = w_old + obs_weight
            self._weight_vol[voxel_index] = w_new
            tsdf_old = self._tsdf_vol[voxel_index]
            self._tsdf_vol[voxel_index] = (tsdf_old * w_old + obs_weight * dists[i]) / w_new

    def find_relevant_voxels_gpu_mode(self, point_cloud):
        point_cloud_x = np.array(point_cloud[:, 0]).astype(np.float32)
        point_cloud_y = np.array(point_cloud[:, 1]).astype(np.float32)

        # prepare output arrays
        voxels_x = np.zeros(len(point_cloud)).astype(np.int32)
        voxels_y = np.zeros(len(point_cloud)).astype(np.int32)

        self.init_gpu_grid(len(point_cloud))
        for gpu_loop_idx in range(self._n_gpu_loops):
            self._cuda_find_voxels(cuda.InOut(self._vol_origin.astype(np.float32)),
                                   cuda.InOut(np.asarray([
                                       gpu_loop_idx,
                                       self._voxel_size,
                                       self._trunc_margin
                                   ], np.float32)),
                                   cuda.InOut(point_cloud_x),
                                   cuda.InOut(point_cloud_y),
                                   cuda.InOut(voxels_x),
                                   cuda.InOut(voxels_y),
                                   block=(self._max_gpu_threads_per_block, 1, 1),
                                   grid=self.gpu_grid
                                   )
        return voxels_x, voxels_y

    def init_gpu_grid(self, n_points):
        gpu_dev = cuda.Device(0)
        self._max_gpu_threads_per_block = gpu_dev.MAX_THREADS_PER_BLOCK
        n_blocks_required = int(np.ceil(float(n_points) / float(self._max_gpu_threads_per_block)))
        self.init_gpu_grid_dim_from_n_blocks_required(gpu_dev, n_blocks_required)
        n_blocks_per_loop = np.prod(self._max_gpu_grid_dim)
        n_threads = n_blocks_per_loop * self._max_gpu_threads_per_block
        self._n_gpu_loops = int(np.ceil(float(n_points) / float(n_threads)))

    def init_gpu_grid_dim_from_n_blocks_required(self, gpu_dev, n_blocks_required):
        remaining_required_blocks = float(n_blocks_required)
        grid_dim_x = int(np.floor(np.cbrt(remaining_required_blocks)))
        grid_dim_x = min(gpu_dev.MAX_GRID_DIM_X, grid_dim_x)

        remaining_required_blocks /= float(grid_dim_x)
        grid_dim_y = int(np.floor(np.sqrt(remaining_required_blocks)))
        grid_dim_y = min(gpu_dev.MAX_GRID_DIM_Y, grid_dim_y)

        remaining_required_blocks /= float(grid_dim_y)
        grid_dim_z = int(np.ceil(remaining_required_blocks))
        grid_dim_z = min(gpu_dev.MAX_GRID_DIM_Z, grid_dim_z)

        self._max_gpu_grid_dim = np.array([grid_dim_x, grid_dim_y, grid_dim_z]).astype(int)
        self.gpu_grid = int(self._max_gpu_grid_dim[0]), int(self._max_gpu_grid_dim[1]), int(self._max_gpu_grid_dim[2])

    def get_volume(self):
        return self._tsdf_vol

    def get_point_cloud(self):
        """Extract a point cloud from the voxel volume.
    """
        tsdf_vol = self.get_volume()

        # Marching cubes
        verts = measure._marching_cubes_lewiner.marching_cubes(tsdf_vol, level=0.2)[0]
        verts = verts * self._voxel_size + self._vol_origin

        colors = np.ones(verts.shape) * 255

        pc = np.hstack([verts, colors])
        return pc

    def get_mesh(self):
        """Compute a mesh from the voxel volume using marching cubes.
    """
        tsdf_vol = self.get_volume()

        # Marching cubes
        verts, faces, norms, vals = measure._marching_cubes_lewiner.marching_cubes(tsdf_vol, level=0.2)
        verts = verts * self._voxel_size + self._vol_origin  # voxel grid coordinates to world coordinates

        colors = np.ones(verts.shape) * 255
        return verts, faces, norms, colors


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) point cloud.
  """
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image
  """
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array([
        (np.array([0, 0, im_w, im_w, 0, 0, im_w, im_w]) - cam_intr[0, 2]) *
        cam_intr[0, 0],
        (np.array([0, 0, 0, 0, im_h, im_h, im_h, im_h]) - cam_intr[1, 2]) *
        cam_intr[1, 1],
        np.array([0, max_depth, 0, max_depth, 0, max_depth, 0, max_depth])
    ])
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def get_vol_bnds(point_clouds):
    point_clouds = np.vstack(point_clouds)
    vol_bnds = np.zeros((3, 2))
    vol_bnds[:, 0] = np.min(point_clouds, axis=0)
    vol_bnds[:, 1] = np.max(point_clouds, axis=0)
    return vol_bnds


def get_point_3d_from_depth_pixel(depth_im, cam_intr, cam_pose, x, y):
    point_3d = np.array([
        (x - cam_intr[0, 2]) * cam_intr[0, 0],
        (y - cam_intr[1, 2]) * cam_intr[1, 1],
        depth_im[x, y],
        1
    ])
    point_3d = np.dot(cam_pose, point_3d)
    return point_3d[:3]


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file.
  """
    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write("%f %f %f %f %f %f %d %d %d\n" % (
            verts[i, 0], verts[i, 1], verts[i, 2],
            norms[i, 0], norms[i, 1], norms[i, 2],
            colors[i, 0], colors[i, 1], colors[i, 2],
        ))

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
  """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, 'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n" % (
            xyz[i, 0], xyz[i, 1], xyz[i, 2],
            rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))
