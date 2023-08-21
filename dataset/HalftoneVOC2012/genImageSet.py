import cv2
import mmcv
import os
import numpy as np
import math
import argparse
import subprocess

# Path to original VOC2012 dataset
input_dir = "/research/d4/gds/cklau21/Datasets/voc2012/VOCdevkit/VOC2012/JPEGImages"

if not os.path.exists(input_dir):
  raise RuntimeError(f"Cannot find input directory {input_dir}, please update the variable input_dir to point to the JPEGImages directory of VOC2012")

output_root_dir = 'HalftoneVOC2012'
input_files = [ f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)) ]

training_set_boundary = 13758
is_training_set = lambda i : (i < training_set_boundary)

max_batch = 1

if not os.path.exists(os.path.join(os.curdir, output_root_dir)):
    os.mkdir(os.path.join(os.curdir, output_root_dir))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'train'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'train', 'raw_ov'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'train', 'target_c'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'val'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'val', 'raw_ov'))
    os.mkdir(os.path.join(os.curdir, output_root_dir, 'val', 'target_c'))  

def dither_and_save(i):
  input_file = input_files[i]
  raw_input_img = cv2.imread(os.path.join(input_dir, input_file), cv2.IMREAD_COLOR)

  # Crop and resize image into 256x256
  height, width, channel = raw_input_img.shape
  w = height if height < width else width
  w = math.floor(w/2)
  center =  (math.floor(width / 2), math.floor(height / 2))
  box = np.array([ center[0] - w, center[1] - w, center[0] + w -1, center[1] + w - 1 ])
  cropped_img = mmcv.imcrop(raw_input_img, box)
  resized_img = mmcv.imresize(cropped_img, (256, 256), return_scale=False)
  gray_img = mmcv.bgr2gray(resized_img)

  mmcv.imwrite(gray_img, 'tmp/gray.pgm')
  subprocess.call(["./ov_dither", "tmp/gray.pgm", "tmp/ov.pgm", "256"])
  ov_halftone_img = mmcv.imread("tmp/ov.pgm", 'grayscale')

  if is_training_set(i):
    output_dir = os.path.join(os.path.curdir, output_root_dir, 'train')
    name = format(i + 1, '05')
    mmcv.imwrite(resized_img, os.path.join(output_dir, 'target_c', name + '.png'))
    # mmcv.imwrite(halftone_img, os.path.join(output_dir, 'raw_ov', name + '.png'))
    mmcv.imwrite(ov_halftone_img, os.path.join(output_dir, 'raw_ov', name + '.png'))
    print(f"saved train target_c & raw_ov image {name}.png")
  else:
    output_dir = os.path.join(os.path.curdir, output_root_dir, 'val')
    name = format(i + 1 - training_set_boundary, '05')
    mmcv.imwrite(resized_img, os.path.join(output_dir, 'target_c', name + '.png'))
    # mmcv.imwrite(halftone_img, os.path.join(output_dir, 'raw_ov', name + '.png'))
    mmcv.imwrite(ov_halftone_img, os.path.join(output_dir, 'raw_ov', name + '.png'))
    print(f"saved val target_c & raw_ov image {name}.png")

def main(batch):
  
  batch_size = math.ceil((len(input_files) / max_batch))
  start = batch_size * (batch - 1)
  end =  min(batch_size * (batch), len(input_files))
  print(f'start: {start}, end: {end}, batch_size:{end - start}')

  for i in range(start, end):
    dither_and_save(i)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Gen image in batches')
  parser.add_argument('--batch', type=int, default=1, required=False, help='the batch number')
  args = parser.parse_args()
  main(args.batch)












