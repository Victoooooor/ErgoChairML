import tensorflow as tf
from matplotlib import pyplot as plt
from IPython import display

import os
import time
import datetime

class pix2pix:

  def __init__(self, cpdir, Buf_S=400, Bat_S=32, IW=256, IH=256, Lambda=100, ldr="logs/"):
    #Define HyperParams
    self.BUFFER_SIZE = Buf_S
    self.BATCH_SIZE = Bat_S
    #Match Dataset Dims
    self.IMG_WIDTH = IW
    self.IMG_HEIGHT = IH
    self.OUTPUT_CHANNELS = 3
    self.LAMBDA = Lambda #Loss function Scalar

    self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    self.generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    self.checkpoint_dir = cpdir
    self.checkpoint_prefix = "ckpt"
    self.log_dir=ldr
    self.generator = self.Generator()
    self.discriminator = self.Discriminator()
    # tf.keras.utils.plot_model(self.generator, show_shapes=True, dpi=64)
    self.summary_writer = tf.summary.create_file_writer(
    self.log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

  def load(self,image_file):
    # Read and decode an image file to a uint8 tensor
    image = tf.io.read_file(image_file)
    image = tf.io.decode_jpeg(image)

    # Split each image tensor into two tensors:
    # - one with a real building facade image
    # - one with an architecture label image
    w = tf.shape(image)[1]
    w = w // 2
    input_image = image[:, w:, :]
    real_image = image[:, :w, :]

    # Convert both images to float32 tensors
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)

    return input_image, real_image

  def resize(self,input_image, real_image, height, width):
    input_image = tf.image.resize(input_image, [height, width],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    real_image = tf.image.resize(real_image, [height, width],
                                 method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return input_image, real_image

  def random_crop(self,input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image, size=[2, self.IMG_HEIGHT, self.IMG_WIDTH, 3])

    return cropped_image[0], cropped_image[1]

  def normalize(self,input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1

    return input_image, real_image

  @tf.function()
  def random_jitter(self,input_image, real_image):
    # Resizing to 286x286
    input_image, real_image = self.resize(input_image, real_image, 286, 286)

    # Random cropping back to 256x256
    input_image, real_image = self.random_crop(input_image, real_image)

    if tf.random.uniform(()) > 0.5:
      # Random mirroring
      input_image = tf.image.flip_left_right(input_image)
      real_image = tf.image.flip_left_right(real_image)

    return input_image, real_image

  def load_image_train(self,image_file):
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.random_jitter(input_image, real_image)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image

  def load_image_test(self,image_file):
    input_image, real_image = self.load(image_file)
    input_image, real_image = self.resize(input_image, real_image,
                                     self.IMG_HEIGHT, self.IMG_WIDTH)
    input_image, real_image = self.normalize(input_image, real_image)

    return input_image, real_image

  def downsample(self, filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                               kernel_initializer=initializer, use_bias=False))

    if apply_batchnorm:
      result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

  def upsample(self, filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

  def Generator(self):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])

    down_stack = [
      self.downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
      self.downsample(128, 4),  # (batch_size, 64, 64, 128)
      self.downsample(256, 4),  # (batch_size, 32, 32, 256)
      self.downsample(512, 4),  # (batch_size, 16, 16, 512)
      self.downsample(512, 4),  # (batch_size, 8, 8, 512)
      self.downsample(512, 4),  # (batch_size, 4, 4, 512)
      self.downsample(512, 4),  # (batch_size, 2, 2, 512)
      self.downsample(512, 4),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
      self.upsample(512, 4, apply_dropout=True),  # (batch_size, 2, 2, 1024)
      self.upsample(512, 4, apply_dropout=True),  # (batch_size, 4, 4, 1024)
      self.upsample(512, 4, apply_dropout=True),  # (batch_size, 8, 8, 1024)
      self.upsample(512, 4),  # (batch_size, 16, 16, 1024)
      self.upsample(256, 4),  # (batch_size, 32, 32, 512)
      self.upsample(128, 4),  # (batch_size, 64, 64, 256)
      self.upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(self.OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
      x = down(x)
      skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
      x = up(x)
      x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

  def generator_loss(self,disc_generated_output, gen_output, target):
    gan_loss = self.loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # Mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (self.LAMBDA * l1_loss)

    return total_gen_loss, gan_loss, l1_loss

  def Discriminator(self):
    initializer = tf.random_normal_initializer(0., 0.02)

    inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

    x = tf.keras.layers.concatenate([inp, tar])  # (batch_size, 256, 256, channels*2)

    down1 = self.downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = self.downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = self.downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

    last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp, tar], outputs=last)

  def discriminator_loss(self,disc_real_output, disc_generated_output):
    real_loss = self.loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = self.loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

  def generate_images(self, model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
      plt.subplot(1, 3, i+1)
      plt.title(title[i])
      # Getting the pixel values in the [0, 1] range to plot.
      plt.imshow(display_list[i] * 0.5 + 0.5)
      plt.axis('off')
    plt.show()

  @tf.function
  def train_step(self, input_image, target, step):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      gen_output = self.generator(input_image, training=True)

      disc_real_output = self.discriminator([input_image, target], training=True)
      disc_generated_output = self.discriminator([input_image, gen_output], training=True)

      gen_total_loss, gen_gan_loss, gen_l1_loss = self.generator_loss(disc_generated_output, gen_output, target)
      disc_loss = self.discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss,
                                            self.generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 self.discriminator.trainable_variables)

    self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                            self.generator.trainable_variables))
    self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                self.discriminator.trainable_variables))

    with self.summary_writer.as_default():
      tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
      tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
      tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
      tf.summary.scalar('disc_loss', disc_loss, step=step//1000)

  def fit(self,train_ds, test_ds, steps):
    example_input, example_target = next(iter(test_ds.take(1)))
    start = time.time()

    for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
      if (step) % 1000 == 0:
        display.clear_output(wait=True)

        if step != 0:
          print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

        start = time.time()

        self.generate_images(self.generator, example_input, example_target)
        print(f"Step: {step//1000}k")

      self.train_step(input_image, target, step)

      # Training step
      if (step+1) % 10 == 0:
        print('.', end='', flush=True)


      # Save (checkpoint) the model every 5k steps
      if (step + 1) % 5000 == 0:
        self.manager.save()
    self.manager.save()

  def train(self, train_dataset, test_dataset, steps=10000):

    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                     discriminator_optimizer=self.discriminator_optimizer,
                                     generator=self.generator,
                                     discriminator=self.discriminator)
    self.manager = tf.train.CheckpointManager(self.checkpoint,self.checkpoint_dir,
                                              max_to_keep=3, checkpoint_name=self.checkpoint_prefix)
    # print(list(PATH.iterdir()))

    self.fit(train_dataset, test_dataset, steps=steps)
    #for demo, gen 5 examples
    for inp, tar in test_dataset.take(5):
      self.generate_images(self.generator, inp, tar)

  def loadcp(self):
    self.checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
                                          discriminator_optimizer=self.discriminator_optimizer,
                                          generator=self.generator,
                                          discriminator=self.discriminator)
    self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir,
                                          max_to_keep=3, checkpoint_name=self.checkpoint_prefix)
    self.checkpoint.restore(self.manager.latest_checkpoint).expect_partial()
    if self.manager.latest_checkpoint:
      print("Restored from {}".format(self.manager.latest_checkpoint))
    else:
      print("No checkpoints loaded, new training session")
