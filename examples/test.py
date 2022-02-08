import os
import pathlib

import tensorflow as tf
from PIL import Image, ImageSequence
from gen_chair.gen_multi import Preprocess
from gen_chair import pix2pix
from gen_chair import coco

class InferenceConfig(coco.CocoConfig):
  # Set batch size to 1 since we'll be running inference on
  # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

def Infer(preproc, generator, origin, masked = True):

  pre = preproc(origin)
  if type(pre) is tuple:
      if masked:
        seg = pre[0]
      else:
        seg = pre[1]
  else:
      seg = None
  origin = tf.cast(tf.image.resize_with_pad(origin, 256, 256), dtype=tf.int32)

  if seg is not None:
    seg  = tf.keras.preprocessing.image.img_to_array(seg)

  else:
    seg = origin

  seg = tf.expand_dims(seg, axis=0)
  seg = tf.cast(tf.image.resize_with_pad(seg, 256, 256), dtype=tf.int32)

  gen = generator(seg ,training=True)
  gen = tf.keras.utils.array_to_img(gen[0])

  seg = tf.keras.utils.array_to_img(seg[0])

  origin = tf.keras.utils.array_to_img(origin)
  dst = Image.new('RGB', (origin.width + seg.width + gen.width, origin.height))
  dst.paste(origin, (0, 0))
  dst.paste(seg, (origin.width, 0))
  dst.paste(gen, (origin.width+seg.width, 0))
  return dst

print(os.getcwd())


# PATH = '../pose'
# path_to_zip = pathlib.Path(PATH)
# PATH = path_to_zip / 'sitting_in_chair'

cp_path = ''
cpdir = '../masked_checkpoints'

# image_names = sorted(
#     [n for n in os.listdir(PATH) if not n.startswith('.')])

prep = Preprocess(InferenceConfig)
cpdir_mask = '../masked_checkpoints'
masked = pix2pix(cpdir_mask)
masked.loadcp()

cpdir_ske = '../skeleton_checkpoints'
skeleton = pix2pix(cpdir_ske)
skeleton.loadcp()
outdr1 = '../out_seg'
outdr2 = '../out_ske'
img_suffix = ('.jpg', '.png', '.ico', '.gif', '.jpeg')
vid_suffix = ('.avi','.mp4')
os.makedirs(outdr1,exist_ok=True)
os.makedirs(outdr2,exist_ok=True)

import numpy as np
import cv2

PATH = '../data/gif'
# PATH = '../data/cartoon_sitting'
imgdr = PATH
image_name = 'tt.mp4'
# image_name = 'cartoon sitting_3.jpeg'
image_path = os.path.join(imgdr, image_name)
if image_path.endswith('.gif'):
    image = Image.open(image_path)
    image.seek(0)
    mask_frames = []
    ske_frames = []
    durations = []
    for frame in ImageSequence.Iterator(image):
        durations.append(image.info['duration'])
        frame = frame.convert('RGB')
        tfframe = tf.convert_to_tensor(np.array(frame))
        mframe = Infer(prep.img_seg, masked.generator, tfframe, True)
        sframe = Infer(prep.img_seg, skeleton.generator, tfframe, False)

        mask_frames.append(mframe)
        ske_frames.append(sframe)
    mask_frames[0].save(os.path.join(outdr1, image_name),
        save_all=True,
        append_images=mask_frames[1:],
        duration=durations,
        loop=0)
    ske_frames[0].save(os.path.join(outdr2, image_name),
        save_all=True,
        append_images=ske_frames[1:],
        duration=durations,
        loop=0)
elif image_path.endswith(img_suffix):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image)
    print(type(image))
    maskout = Infer(prep.img_seg, masked.generator, image, True)
    skeletonout = Infer(prep.img_seg, skeleton.generator, image, False)
    maskout.save(os.path.join(outdr1, image_name))
    skeletonout.save(os.path.join(outdr2, image_name))
elif image_path.endswith(vid_suffix):
    print('video',image_path)
    video = cv2.VideoCapture(image_path)
    if not video.isOpened():
        print('video does not exist')
        exit()
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_num = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = 256*3
    frame_height = 256
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    name1 = os.path.join(outdr1, image_name)
    name2 = os.path.join(outdr2, image_name)
    video_writer1 = cv2.VideoWriter(name1,fourcc,fps,(frame_width,frame_height))
    video_writer2 = cv2.VideoWriter(name2,fourcc,fps,(frame_width,frame_height))
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
    counter =0
    while True:
      ret, frame = video.read()
      if ret == True:
          tfframe= tf.convert_to_tensor(frame)
          mframe = Infer(prep.img_seg, masked.generator, tfframe, True)
          sframe = Infer(prep.img_seg, skeleton.generator, tfframe, False)
          m_cv2 = cv2.cvtColor(np.array(mframe), cv2.COLOR_RGB2BGR)
          s_cv2 = cv2.cvtColor(np.array(sframe), cv2.COLOR_RGB2BGR)
          video_writer1.write(m_cv2)
          video_writer2.write(s_cv2)
          counter +=1
      if ret == False:
          break
    video.release()
    video_writer1.release()
    video_writer2.release()
    cv2.destroyAllWindows()