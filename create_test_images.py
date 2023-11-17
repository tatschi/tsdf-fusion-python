import os
import numpy as np
import open3d as o3d

DATA_PATH = "data/test-images"
FRAME_FILENAME = lambda x: os.path.join(DATA_PATH, "frame-%02d.ply" % x)


def main():
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
    o3d.io.write_point_cloud(FRAME_FILENAME(1), pc_o3d)


if __name__ == "__main__":
    main()
