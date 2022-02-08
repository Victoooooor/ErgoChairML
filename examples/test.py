import os
import pathlib

import tensorflow as tf
from PIL import Image, ImageSequence
from gen_chair.gen_multi import Preprocess
from gen_chair.gen_pose import Preprocess as Preprocess2
from gen_chair import pix2pix
from gen_chair import coco

class InferenceConfig(coco.CocoConfig):
  # Set batch size to 1 since we'll be running inference on
  # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
  GPU_COUNT = 1
  IMAGES_PER_GPU = 1

def Infer(preproc, generator, origin):
    seg = preproc(origin)
    origin = tf.cast(tf.image.resize_with_pad(origin, 256, 256), dtype=tf.int32)

    if seg is not None:
        seg = tf.keras.preprocessing.image.img_to_array(seg)

    else:
        seg = origin

    seg = tf.expand_dims(seg, axis=0)
    seg = tf.cast(tf.image.resize_with_pad(seg, 256, 256), dtype=tf.int32)

    gen = generator(seg, training=True)
    gen = tf.keras.utils.array_to_img(gen[0])

    seg = tf.keras.utils.array_to_img(seg[0])

    origin = tf.keras.utils.array_to_img(origin)
    dst = Image.new('RGB', (origin.width + seg.width + gen.width, origin.height))
    dst.paste(origin, (0, 0))
    dst.paste(seg, (origin.width, 0))
    dst.paste(gen, (origin.width + seg.width, 0))
    return dst

if __name__ == "__main__":

    print(os.getcwd())
    PATH = '../data/gif'
    # PATH = '../pose'
    path_to_zip = pathlib.Path(PATH)
    PATH = path_to_zip / 'sitting_in_chair'

    cp_path = ''
    cpdir = '../masked_checkpoints'

    # image_names = sorted(
    #     [n for n in os.listdir(PATH) if not n.startswith('.')])
    imgdr = PATH
    image_name = 'sitting in chair_13.png'

    prep = Preprocess(InferenceConfig)
    prep2 = Preprocess2()
    cpdir_mask = '../masked_checkpoints'
    masked = pix2pix(cpdir_mask)
    masked.loadcp()

    cpdir_ske = '../skeleton_checkpoints'
    skeleton = pix2pix(cpdir_ske)
    skeleton.loadcp()
    outdr1 = '../out_seg'
    outdr2 = '../out_ske'
    img_suffix = ('.jpg', '.png', '.ico', '.gif', '.jpeg')
    image_path = os.path.join(imgdr, image_name)
    os.mkdir(outdr1,exist_ok=True)
    os.mkdir(outdr2,exist_ok=True)
    if image_path.endswith('.gif'):
        image = Image.open(image_path)
        image.seek(0)
        mask_frames = []
        ske_frames = []
        durations = []
        for frame in ImageSequence.Iterator(image):
            durations.append(image.info['duration'])
            frame = frame.convert('RGB')
            tfframe = tf.keras.preprocessing.image.img_to_array(frame)
            mframe = Infer(prep.img_seg, masked.generator, tfframe)
            sframe = Infer(prep2.img_seg, skeleton.generator, tfframe)




    # p2p = pix2pix(cpdir)
    # p2p.loadcp()
    # try:
    #     test_dataset = tf.data.Dataset.list_files(str(PATH / 'test*.*'))
    # except tf.errors.InvalidArgumentError:
    #     test_dataset = tf.data.Dataset.list_files(str(PATH / 'val*.*'))
    # test_dataset = test_dataset.map(p2p.load_image_test)
    # test_dataset = test_dataset.batch(p2p.BATCH_SIZE)
    # for inp, tar in test_dataset.take(2):
    #   p2p.generate_images(p2p.generator, inp, tar)
