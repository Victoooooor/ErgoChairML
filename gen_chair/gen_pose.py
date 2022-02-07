import copy
import csv
import numpy as np
import pandas as pd
import os
import sys
import tempfile
import tqdm
import shutil
from matplotlib import pyplot as plt
from PIL import Image

import tensorflow as tf

# Load model from local, tf_hub did not for w/ cuDNN 8.3



# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.




# Load MoveNet Thunder model
from . import utils
from .ml import Movenet
from .data import BodyPart

class Preprocess(object):
  """Helper class to preprocess pose sample images for classification."""

  def __init__(self):
    """Preprocessing: generate corresponding poseture image for valid image
    """
    self._messages = []
    # Create a temp dir to store the pose CSVs per class
    self._csvs_out_folder_per_class = tempfile.mkdtemp()

    # Get list of pose classes and print image statistics
    self.movenet = Movenet('movenet_thunder')
    self.detection_threshold = 0.1

  def get_concat_h(self,im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

  def detect(self, input_tensor, inference_count=3):
    """Runs detection on an input image.
    """
    image_height, image_width, channel = input_tensor.shape
    # Detect pose using the full input image
    try:
      self.movenet.detect(input_tensor.numpy(), reset_crop_region=True)

      # Repeatedly using previous detection result to identify the region of
      # interest and only croping that region to improve detection accuracy
      for _ in range(inference_count - 1):
        person = self.movenet.detect(input_tensor.numpy(),
                                reset_crop_region=False)

      return person
    except:
      return None

  def draw_prediction_on_image(
          self, image, person, crop_region=None, close_figure=True,
          keep_input_size=False):
    """Draws the keypoint predictions on image.
    """
    # Draw the detection result on black canvas of same size.
    image_np = utils.visualize(image, [person], 100)

    # Plot the image with detection results.
    if not close_figure:
      height, width, channel = image.shape
      aspect_ratio = float(width) / height
      fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
      im = ax.imshow(image_np)
      # plt.close(fig)
    # print(type(image_np))

    if not keep_input_size:
      PIL_image = Image.fromarray(image_np.astype('uint8'), 'RGB')
      resized_image = PIL_image.resize((256, 256))
      return resized_image

    return image_np

  def process(self,
              images_folder,
              pose_folder,
              csvs_out_path = './',
              per_class_limit=None,
              detection_threshold=0.1,
              split = 0.7):
    """Preprocesses images in the given folder.
    """
    self.split = split
    self._images_folder = images_folder
    self._pose_folder = pose_folder
    self._pose_class_names = sorted(
      [n for n in os.listdir(images_folder) if not n.startswith('.')
       if os.path.isdir(os.path.join(self._images_folder, n))]
    )
    if len(self._pose_class_names) == 0:
      raise FileNotFoundError
    if detection_threshold is not None:
      self.detection_threshold= detection_threshold
    # Loop through the classes and preprocess its images
    for pose_class_name in self._pose_class_names:
      print('Preprocessing:', pose_class_name, file=sys.stderr)

      # Paths for the pose class.
      images_folder = os.path.join(self._images_folder, pose_class_name)
      pose_folder = os.path.join(self._pose_folder, pose_class_name)
      csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  pose_class_name + '.csv')

      if not os.path.exists(pose_folder):
        os.makedirs(pose_folder)

      # Detect landmarks in each image and write it to a CSV file
      with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file,
                                    delimiter=',',
                                    quoting=csv.QUOTE_MINIMAL)
        # Get list of images
        image_names = sorted(
            [n for n in os.listdir(images_folder) if not n.startswith('.')])
        if per_class_limit is not None:
          image_names = image_names[:per_class_limit]

        valid_image_count = 0

        # Detect pose landmarks from each image
        for image_name in tqdm.tqdm(image_names):
          image_path = os.path.join(images_folder, image_name)

          try:
            image = tf.io.read_file(image_path)
            image = tf.io.decode_jpeg(image)
            image_height, image_width, channel = image.shape
          except:
            self._messages.append('Skipped and removed' + image_path + '. Invalid image.')
            os.remove(image_path)
            continue

          # Skip images that isn't RGB because Movenet requires RGB images
          if channel != 3:
            self._messages.append('Skipped and removed' + image_path +
                                  '. Image isn\'t in RGB format.')
            os.remove(image_path)
            continue
          person = self.detect(image)

          # Save landmarks if all landmarks were detected
          min_landmark_score = min(
              [keypoint.score for keypoint in person.keypoints])
          should_keep_image = min_landmark_score >= self.detection_threshold
          if not should_keep_image:
            self._messages.append('Skipped and removed' + image_path +
                                  '. No pose was confidentlly detected.')
            os.remove(image_path)
            continue

          valid_image_count += 1

          # Draw the prediction result on top of the image for debugging later
          output_overlay = self.draw_prediction_on_image(np.full_like(image.numpy(),0), person,
              close_figure=True, keep_input_size=False)

          # Write detection result into an image file
          PIL_image = Image.fromarray(image.numpy().astype('uint8'), 'RGB')
          resized_image = PIL_image.resize((256, 256))
          concated = self.get_concat_h(resized_image, output_overlay)
          # print(type(concated))
          # output_frame = cv2.cvtColor(concated, cv2.COLOR_RGB2BGR)
          # cv2.imwrite(os.path.join(pose_folder, image_name), output_frame)
          if np.random.uniform() < self.split:
            concated.save(os.path.join(pose_folder, 'train_'+image_name))
          else:
            concated.save(os.path.join(pose_folder, 'test_' + image_name))
          # Get landmarks and scale it to the same size as the input image
          pose_landmarks = np.array(
              [[keypoint.coordinate.x, keypoint.coordinate.y, keypoint.score]
                for keypoint in person.keypoints],
              dtype=np.float32)

          # Write the landmark coordinates to its per-class CSV file
          coordinates = pose_landmarks.flatten().astype(str).tolist()
          csv_out_writer.writerow([image_name] + coordinates)

        if not valid_image_count:
          raise RuntimeError(
              'No valid images found for the "{}" class.'
              .format(pose_class_name))

    # Print the error message collected during preprocessing.
    print('\n'.join(self._messages))

    # Combine all per-class CSVs into a single output file

    all_landmarks_df = self._all_landmarks_as_dataframe()
    all_landmarks_df.to_csv(os.path.join(csvs_out_path,
               'landmarks_df' + '.csv'), index=False)

  def img_seg(self, image_path):
    try:
      image = tf.io.read_file(image_path)
      image = tf.io.decode_jpeg(image)
      image = tf.expand_dims(image, axis=0)
      image_origin = copy.copy(image)
      image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
      _, image_height, image_width, channel = image_origin.shape
    except:
      self._messages.append('Skipped' + image_path + '. Invalid image.')
      return None

    # Skip images that isn't RGB because Movenet requires RGB images
    if channel != 3:
      self._messages.append('Skipped' + image_path +
                            '. Image isn\'t in RGB format.')
      return None

    person = self.detect(image)

    # Save landmarks if all landmarks were detected
    min_landmark_score = min(
      [keypoint.score for keypoint in person.keypoints])
    should_keep_image = min_landmark_score >= self.detection_threshold
    if not should_keep_image:
      self._messages.append('Skipped and removed' + image_path +
                            '. No pose was confidentlly detected.')

      return None
    output_overlay = self.draw_prediction_on_image(np.full_like(image.numpy(), 0), person,
                                                   close_figure=True, keep_input_size=False)

    return output_overlay

  def class_names(self):
    """List of classes found in the training dataset."""
    return self._pose_class_names

  def _all_landmarks_as_dataframe(self):
    """Merge all per-class CSVs into a single dataframe."""
    total_df = None
    for class_index, class_name in enumerate(self._pose_class_names):
      csv_out_path = os.path.join(self._csvs_out_folder_per_class,
                                  class_name + '.csv')
      per_class_df = pd.read_csv(csv_out_path, header=None)

      # Add the labels
      per_class_df['class_no'] = [class_index]*len(per_class_df)
      per_class_df['class_name'] = [class_name]*len(per_class_df)

      # Append the folder name to the filename column (first column)
      per_class_df[per_class_df.columns[0]] = (os.path.join(class_name, '')
        + per_class_df[per_class_df.columns[0]].astype(str))

      if total_df is None:
        # For the first class, assign its data to the total dataframe
        total_df = per_class_df
      else:
        # Concatenate each class's data into the total dataframe
        total_df = pd.concat([total_df, per_class_df], axis=0)

    list_name = [[bodypart.name + '_x', bodypart.name + '_y',
                  bodypart.name + '_score'] for bodypart in BodyPart]
    header_name = []
    for columns_name in list_name:
      header_name += columns_name
    header_name = ['file_name'] + header_name
    header_map = {total_df.columns[i]: header_name[i]
                  for i in range(len(header_name))}

    total_df.rename(header_map, axis=1, inplace=True)

    return total_df
