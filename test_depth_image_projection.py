import cv2
import numpy as np
import os
import fusion
import open3d as o3d

DATA_PATH = "data/2023-07-10"
CAM_INTR_PATH = os.path.join(DATA_PATH, "camera-intrinsics.txt")
DEPTH_PATH = lambda x: os.path.join(DATA_PATH, "depth/frame-%02d.png" % x)
CAM_POSE_PATH = lambda x: os.path.join(DATA_PATH, "pose/frame-%02d.pose.txt" % x)


# def pixel_to_3d(depth_im, cam_intr, cam_pose, x, y):
#     point_3d = np.array([
#         (x - cam_intr[0, 2]) * cam_intr[0, 0],
#         (y - cam_intr[1, 2]) * cam_intr[1, 1],
#         depth_im[x, y],
#         1
#     ])
#
#     point_3d = np.dot(cam_pose, point_3d)
#     return point_3d[:3]

def pixel_to_3d(depth_im, cam_intr, cam_pose, x, y):
    point_3d = np.array([
        (y - cam_intr[1, 2]) * cam_intr[1, 1],
        (x - cam_intr[0, 2]) * cam_intr[0, 0],
        depth_im[x, y],
        1
    ])
    # point_3d = np.array([y, x, depth_im[x, y], 1])

    # pose = cam_pose.copy()
    #pose = np.linalg.inv(pose)
    # pose[[0, 1]] = pose[[1, 0]]
    # pose[:, [0, 1]] = pose[:, [1, 0]]
    # point_3d = np.dot(pose, point_3d)
    return point_3d[:3]


def main():
    cam_intr = np.loadtxt(CAM_INTR_PATH, delimiter=' ')
    pc = []
    for i in range(1):
        depth_im = cv2.imread(DEPTH_PATH(i+1), -1).astype(float)
        depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
        depth_im[depth_im == 65.535] = 0
        cam_pose = np.loadtxt(CAM_POSE_PATH(i+1))  # 4x4 rigid transformation matrix

        pc_from_image = [pixel_to_3d(depth_im, cam_intr, cam_pose, x, y)
                         for x in range(depth_im.shape[0])
                         for y in range(depth_im.shape[1])]
        pc.extend(pc_from_image)
    return pc


if __name__ == "__main__":
    pointcloud = main()
    pc_o3d = o3d.geometry.PointCloud()
    pc_o3d.points = o3d.utility.Vector3dVector(pointcloud)
    o3d.io.write_point_cloud("test_depth_image_projection.ply", pc_o3d)
