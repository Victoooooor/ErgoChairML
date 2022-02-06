import os

import skimage
import random

from gen_chair import coco
from gen_chair import segmentation
from PIL import Image

import tensorflow_hub as hub
import tensorflow as tf
IMAGE_DIR = "/home/victor/Desktop/img2img/data/sitting_in_chair/"


if __name__ == "__main__":
    images_folder= './data/sitting/'
    image_names = sorted(
        [n for n in os.listdir(images_folder) if not n.startswith('.')])
    print('loading')
    # detector = hub.load("https://hub.tensorflow.google.cn/tensorflow/mask_rcnn/inception_resnet_v2_1024x1024/1")
    detector = hub.load("https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1")
    print('downloaded')
    for filename in image_names:

        image = tf.io.read_file(os.path.join(images_folder, filename))
        image = tf.io.decode_jpeg(image)
        image = tf.expand_dims(image, axis=0)
        detector_output = detector(image)
        class_ids = detector_output["detection_classes"]
        print(filename)
