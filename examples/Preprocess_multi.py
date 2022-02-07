import os
import shutil
import sys

from gen_chair import coco
from gen_chair.gen_multi import Preprocess
import tensorflow as tf

class InferenceConfig(coco.CocoConfig):
  # Set batch size to 1 since we'll be running inference on
  # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1


if __name__ == '__main__':
  #define meta parameters:
  imgdr = './data'
  posedr = './pose'

  # compare the number of items in data and pose
  # assume dataset is fully paired if the numbers match,
  # only run image preprocessing when not fully paired
  data_count = sum([len(files) for r, d, files in os.walk(imgdr)])
  pose_count = sum([len(files) for r, d, files in os.walk(posedr)])
  if data_count != pose_count:
    try:
      shutil.rmtree(posedr)
    except FileNotFoundError:
      pass
    try:
      prep = Preprocess(InferenceConfig)
    except FileNotFoundError:
      print(f"Image Data Not Found: {imgdr}", file=sys.stderr)
      print('Data Preprocessing skipped')
    # prep.process(imgdr, posedr, None,0.1)
    image = tf.io.read_file('./data/person_sit/person sit_3.jpeg')
    image = tf.io.decode_jpeg(image)
    out = prep.img_seg(image)
    print(type(out))