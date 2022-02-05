import os
import shutil
import sys

from gen_chair.gen_pose import Preprocess




if __name__ == '__main__':
  #define meta parameters:
  imgdr = './datass'
  posedr = './posess'

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
      prep = Preprocess(imgdr,posedr,'./', 0.7)
    except FileNotFoundError:
      print(f"Image Data Not Found: {imgdr}", file=sys.stderr)
      print('Data Preprocessing skipped')
    prep.process(None,0.1)