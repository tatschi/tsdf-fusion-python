import os
import cv2
import numpy as np

DATA_PATH = "data/test-images"
CAM_INTR_PATH = os.path.join(DATA_PATH, "camera-intrinsics.txt")
DEPTH_PATH = lambda x: os.path.join(DATA_PATH, "depth/frame-%02d.png" % x)
CAM_POSE_PATH = lambda x: os.path.join(DATA_PATH, "pose/frame-%02d.pose.txt" % x)


def main():
    depth = np.ones((2, 2)) * 1000
    depth[1, 1] = 2000

    depth = depth.astype(np.uint16)
    cv2.imwrite(DEPTH_PATH(1), depth)


if __name__ == "__main__":
    main()
