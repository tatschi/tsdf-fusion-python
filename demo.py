"""Fuse 1000 depth images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""
import os
import time

import cv2
import numpy as np

import fusion
import open3d as o3d

DATA_PATH = "data/2023-07-10"
CAM_INTR_PATH = os.path.join(DATA_PATH, "camera-intrinsics.txt")
DEPTH_PATH = lambda x: os.path.join(DATA_PATH, "depth/frame-%02d.ply" % x)
CAM_POSE_PATH = lambda x: os.path.join(DATA_PATH, "pose/frame-%02d.pose.txt" % x)

if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all points in all point clouds
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    n_imgs = 1
    pointclouds = []
    for i in range(n_imgs):
        # Read depth image and camera pose
        pc = o3d.io.read_point_cloud(DEPTH_PATH(i+1))
        pc_points = np.asarray(pc.points)
        pointclouds.append(pc_points)

    vol_bnds = fusion.get_vol_bnds(pointclouds)

    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.01)

    # Loop through images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Integrate observation into voxel volume
        tsdf_vol.integrate(pointclouds[i], obs_weight=1.)

    fps = n_imgs / (time.time() - t0_elapse)
    print("Average FPS: {:.2f}".format(fps))

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...")
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite("mesh.ply", verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...")
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite("pc.ply", point_cloud)
