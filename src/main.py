import utils
import cv2 as cv
from pathlib import Path


output_dir_opencv = Path("outputs/opencv")
output_dir_zhang = Path("outputs/opencv")
output_dir_naive = Path("outputs/naive")


def main():
    # the baseline: using opencv to calibrate
    utils.calibrate(cv.calibrateCamera, output_dir=output_dir_opencv)


if __name__ == "__main__":
    main()
