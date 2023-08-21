# This script generate constant color and its corresponding ov_halftone image
#
import mmcv
import os
import numpy as np
import math
import argparse
import subprocess

output_root_dir = 'special'

max_set = 13758 + 3367
training_set_boundary = 13758
def is_training_set(i): return (i < training_set_boundary)


if not os.path.exists(os.path.join(os.curdir, output_root_dir)):
    os.mkdir(os.path.join(os.curdir, output_root_dir))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'train'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'train', 'raw_ov'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'train', 'target_c'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'val'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'val', 'raw_ov'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'val', 'target_c'))


def bgr_img(B: float, G: float, R: float):
    """ value from 0-1 """

    img = np.zeros([256, 256, 3], dtype=np.uint8)
    img[:, :, 0] = np.ones([256, 256]) * math.floor(B * 255)
    img[:, :, 1] = np.ones([256, 256]) * math.floor(G * 255)
    img[:, :, 2] = np.ones([256, 256]) * math.floor(R * 255)
    return img


def dither_and_save(i, img):

    resized_img = img
    gray_img = mmcv.bgr2gray(resized_img)

    mmcv.imwrite(gray_img, 'stmp/gray.pgm')
    subprocess.call(["./ov_dither", "stmp/gray.pgm", "stmp/ov.pgm", "256"])
    ov_halftone_img = mmcv.imread("stmp/ov.pgm", 'grayscale')

    if is_training_set(i):
        output_dir = os.path.join(os.path.curdir, output_root_dir, 'train')
        name = format(i + 1, '05')
    else:
        output_dir = os.path.join(os.path.curdir, output_root_dir, 'val')
        name = format(i + 1 - training_set_boundary, '05')

    mmcv.imwrite(resized_img, os.path.join(output_dir, 'target_c', name + '.png'))
    mmcv.imwrite(ov_halftone_img, os.path.join(output_dir, 'raw_ov', name + '.png'))
    if is_training_set(i):
        print(f"saved train target_c & raw_ov image {name}.png")
    else:
        print(f"saved val target_c & raw_ov image {name}.png")


def main(start, end):
    for i in range(start, end):
        r, g, b = np.random.rand(3)
        img = bgr_img(b, g, r)
        dither_and_save(i, img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Gen image in batches')
    args = parser.parse_args()
    main(start=0, end=max_set)
