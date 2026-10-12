from __future__ import annotations
from pathlib import Path

import argparse
import json

import cv2 as cv
import numpy as np

# reference: https://docs.opencv.org/4.13.0/dc/dbb/tutorial_py_calibration.html

zhang = Path("images/zhang/")
naive = Path("images/naive/")

SQUARE_SIZE_MM = 25.0
PATTERN_SIZE = (10, 7) # interior number of corners


def make_object_points() -> np.ndarray: 
    """ real world coordinates for corners
    The return np.ndarray is the following matrix mutiplied by SQUARE_SIZE

        (0, 0, 0)   (1, 0, 0)   (2, 0, 0)  ...  (9, 0, 0)
        (0, 1, 0)           ......              (9, 1, 0)
                        ......
        (0, 6, 0)           ......              (9, 6, 0)
    """

    # http://numpy.org/doc/stable/reference/generated/numpy.mgrid.html

    cols, rows = PATTERN_SIZE
    points = np.zeros((rows * cols, 3), dtype=np.float32) # (70, 3)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * SQUARE_SIZE_MM

    return points


def read_image(path: Path) -> np.ndarray: 
    flags = cv.IMREAD_COLOR | cv.IMREAD_IGNORE_ORIENTATION
    image = cv.imread(str(path), flags) # (h, w, c), BGR

    if image is None:
        raise RuntimeError(f"fail to load image: {path}")

    return image


def preprocess(
    images_dir: Path,
    output_dir: Path
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    tuple[int, int],
    ]:
    # this function load imaages, find and visualize corners and return points
    images_dir.mkdir(parents=True, exist_ok=True)
    corners_dir = output_dir / "detected_corners" # for visualization
    corners_dir.mkdir(parents=True, exist_ok=True)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []

    image_paths = sorted(images_dir.glob("*.jpg"))
    if not image_paths:
        raise RuntimeError(f"Fail to find JPG files in {images_dir}")

    image_size: tuple[int,int] | None = None
    for path in image_paths:
        image = read_image(path) # # (h, w, 3), uint8
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # (h, w), uint8
        height, weight = gray_image.shape

        if image_size is None:
            image_size = (weight, height)
        elif image_size != (weight, height):
            raise RuntimeError(f"The size of image is wrong")

        found, corners = cv.findChessboardCornersSB(
            image = gray_image,
            patternSize = PATTERN_SIZE,
            flags = cv.CALIB_CB_NORMALIZE_IMAGE,
        ) # bool, (n, 1, 2)

        if found == False or corners is None:
            raise RuntimeError(f"Fail to find the corner for {path.name}")

        object_points.append(make_object_points()) # always the same, (n, 3)
        image_points.append(corners.astype(np.float32)) # (n, 1, 2)

        visualization = image.copy()
        cv.drawChessboardCorners(visualization, 
                                 PATTERN_SIZE, corners, found)
        cv.imwrite(str(corners_dir / path.name), visualization)

    return object_points, image_points, image_size


def reprojection_rmse(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs: tuple[np.ndarray, ...],
    tvecs: tuple[np.ndarray, ...],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> list[float]: 
    """ Calculate reprojection RMSE for each image
    Parameters:
        - object_points: x_w, coordinates under world frame
        - image_points: x_i, coordinates of image plane (pixels)
        - rvecs: R in extrinsic matrix, from world frame to camera frame
        - tvecs: t in extrinsic matrix, from world frame to camera frame
        - camera_matrix: intrinsic matrix K for camera
        - distortion: distortion for each image

    Reproject the real world points into image with given parameters
    and then calculate the RMSE error. If calibration is accuarate, it is small
    """
    errors = []

    for object_pts, observed_pts, rvec, tvec in zip(
        object_points,
        image_points,
        rvecs,
        tvecs,
        strict=True,
    ):
    # https://docs.opencv.org/3.4.20/d9/d0c/group__calib3d.html
        projected_pts, _ = cv.projectPoints(
            object_pts,
            rvec,
            tvec,
            camera_matrix,
            distortion,
        ) # (n, 1, 2)

        residuals = observed_pts.reshape(-1, 2) - projected_pts.reshape(-1, 2)
        rmse = np.sqrt(np.mean(np.sum(residuals ** 2, axis=1)))
        errors.append(float(rmse))

    return errors

def calibrate(func, output_dir):
    object_points, image_points, image_size = preprocess(zhang, output_dir)

    rms, camera_matrix, distortion, rvecs, tvecs = func(
        object_points,
        image_points,
        image_size,
        None,
        None
    )

    rmse = reprojection_rmse(
        object_points = object_points,
        image_points = image_points,
        rvecs = rvecs,
        tvecs = tvecs,
        camera_matrix = camera_matrix,
        distortion = distortion
    )

    print(f"RMS returned: {rms}")
    print(f"Opencv intrinsic matrix:\n{camera_matrix}")
    print(f"Distortion: [k1, k2, p1, p2, k3]:\n{distortion.ravel()}")
    print(f"Reprojection RMSE: {np.array(rmse).mean()}")
