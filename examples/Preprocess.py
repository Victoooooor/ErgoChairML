import os
import shutil
import sys

from gen_chair.gen_pose import Preprocess
import tensorflow as tf



if __name__ == '__main__':
  #define meta parameters:
  imgdr = './data'
  posedr = './poses'

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
      prep = Preprocess()
    except FileNotFoundError:
      print(f"Image Data Not Found: {imgdr}", file=sys.stderr)
      print('Data Preprocessing skipped')
    # prep.process(imgdr,posedr,'./',None,0.1)
    image = tf.io.read_file('./data/person_sit/person sit_3.jpeg')
    image = tf.io.decode_jpeg(image)
    out = prep.img_seg(image)
    print(type(out))