"""Fuse 1000 depth images from the 7-scenes dataset into a TSDF voxel volume with 2cm resolution.
"""
import os
import time

import cv2
import numpy as np

import fusion

DATA_PATH = "data/2023-10-13"
CAM_INTR_PATH = os.path.join(DATA_PATH, "camera-intrinsics.txt")
DEPTH_PATH = lambda x: os.path.join(DATA_PATH, "depth/frame-%02d.png" % x)
CAM_POSE_PATH = lambda x: os.path.join(DATA_PATH, "pose/frame-%02d.pose.txt" % x)

if __name__ == "__main__":
    # ======================================================================================================== #
    # (Optional) This is an example of how to compute the 3D bounds
    # in world coordinates of the convex hull of all camera view
    # frustums in the dataset
    # ======================================================================================================== #
    print("Estimating voxel volume bounds...")
    n_imgs = 1
    cam_intr = np.loadtxt(CAM_INTR_PATH, delimiter=' ')
    vol_bnds = np.zeros((3, 2))
    for i in range(n_imgs):
        # Read depth image and camera pose
        depth_im = cv2.imread(DEPTH_PATH(i+1), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt(CAM_POSE_PATH(i+1))  # 4x4 rigid transformation matrix
        #cam_pose = np.eye(4)

        # Compute camera view frustum and extend convex hull
        view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
        vol_bnds[:, 0] = np.minimum(vol_bnds[:, 0], np.amin(view_frust_pts, axis=1))
        vol_bnds[:, 1] = np.maximum(vol_bnds[:, 1], np.amax(view_frust_pts, axis=1))
    # ======================================================================================================== #

    # ======================================================================================================== #
    # Integrate
    # ======================================================================================================== #
    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.002)

    # Loop through images and fuse them together
    t0_elapse = time.time()
    for i in range(n_imgs):
        print("Fusing frame %d/%d" % (i + 1, n_imgs))

        # Read RGB-D image and camera pose
        depth_im = cv2.imread(DEPTH_PATH(i+1), -1).astype(float)
        depth_im /= 1000.
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt(CAM_POSE_PATH(i+1))
        #cam_pose = np.eye(4)

        # Integrate observation into voxel volume
        tsdf_vol.integrate(depth_im, cam_intr, cam_pose, obs_weight=1.)

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
