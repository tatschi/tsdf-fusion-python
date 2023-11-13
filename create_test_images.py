import os
import cv2
import numpy as np
import open3d as o3d

DATA_PATH = "data/test-images"
CAM_INTR_PATH = os.path.join(DATA_PATH, "camera-intrinsics.txt")
DEPTH_PATH = lambda x: os.path.join(DATA_PATH, "depth/frame-%02d.png" % x)
PC_PATH = lambda x: os.path.join(DATA_PATH, "depth/frame-%02d.ply" % x)
CAM_POSE_PATH = lambda x: os.path.join(DATA_PATH, "pose/frame-%02d.pose.txt" % x)


def main():
    depth = np.ones((2, 2)) * 1000
    depth[1, 1] = 2000

    depth = depth.astype(np.uint16)
    cv2.imwrite(DEPTH_PATH(1), depth)


def pc():
    x_vals = np.linspace(0, 1, 100)
    y_vals = np.linspace(0, 1, 100)
    z_vals = np.zeros(x_vals.shape)
    z_vals[30:60] = 1
    pointcloud = np.zeros((len(x_vals), 3))
    pointcloud[:, 0] = x_vals
    pointcloud[:, 1] = y_vals
    pointcloud[:, 2] = z_vals
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud(PC_PATH(1), pc_o3d)

if __name__ == "__main__":
    #main()
    pc()
