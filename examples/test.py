import os
import pathlib

import tensorflow as tf

from gen_chair import pix2pix

if __name__ == "__main__":
    print(os.getcwd())
    PATH = '../data'
    path_to_zip = pathlib.Path(PATH)
    PATH = path_to_zip / 'sitting_in_chair'

    cp_path = ''
    cpdir = '../checkpoints'
    p2p = pix2pix(cpdir)
    p2p.loadcp()
    image_names = sorted(
        [n for n in os.listdir(PATH) if not n.startswith('.')])
    print(image_names)
    # try:
    #     test_dataset = tf.data.Dataset.list_files(str(PATH / 'test*.*'))
    # except tf.errors.InvalidArgumentError:
    #     test_dataset = tf.data.Dataset.list_files(str(PATH / 'val*.*'))
    # test_dataset = test_dataset.map(p2p.load_image_test)
    # test_dataset = test_dataset.batch(p2p.BATCH_SIZE)
    # for inp, tar in test_dataset.take(2):
    #   p2p.generate_images(p2p.generator, inp, tar)
