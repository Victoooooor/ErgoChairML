import pathlib

import tensorflow as tf

from gen_chair.pix2pix import pix2pix

if __name__ == '__main__':
  PATH = './pose'
  path_to_zip = pathlib.Path(PATH)
  PATH = path_to_zip / 'sitting_in_chair'
  cpdir = './temp_checkpoints'
  p2p = pix2pix(cpdir)
  train_dataset = tf.data.Dataset.list_files(str(PATH / 'train*.*'))
  train_dataset = train_dataset.map(p2p.load_image_train,
                                    num_parallel_calls=tf.data.AUTOTUNE)
  train_dataset = train_dataset.shuffle(p2p.BUFFER_SIZE)
  train_dataset = train_dataset.batch(p2p.BATCH_SIZE)

  try:
    test_dataset = tf.data.Dataset.list_files(str(PATH / 'test*.*'))
  except tf.errors.InvalidArgumentError:
    test_dataset = tf.data.Dataset.list_files(str(PATH / 'val*.*'))
  test_dataset = test_dataset.map(p2p.load_image_test)
  test_dataset = test_dataset.batch(p2p.BATCH_SIZE)
  p2p.train(train_dataset,test_dataset)
  
  # inp, re = p2p.load(str(PATH.parent/'sitting' / 'train_sitting_3.jpeg'))
  # # Casting to int for matplotlib to display the images
  # plt.figure()
  # plt.imshow(inp / 255.0)
  # plt.show()
  # plt.figure()
  # plt.imshow(re / 255.0)
  # plt.show()

  # plt.figure(figsize=(6, 6))
  # for i in range(4):
  #   rj_inp, rj_re = p2p.random_jitter(inp, re)
  #   plt.subplot(2, 2, i + 1)
  #   plt.imshow(rj_inp / 255.0)
  #   plt.axis('off')
  # plt.show()

