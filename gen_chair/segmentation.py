import os
import sys
import random
import math
import numpy as np
import skimage.io

# Root directory of the project
from . import coco
from .mrcnn import utils
from .mrcnn import visualize
from .mrcnn import model as modellib
from PIL import Image
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']


class segmentation:
    def __init__(self,InferenceConfig):
        self.ROOT_DIR = os.getcwd()
        self.MODEL_DIR = os.path.join(self.ROOT_DIR, "logs")
        self.COCO_MODEL_PATH = os.path.join(self.ROOT_DIR, "mask_rcnn_coco.h5")
        if not os.path.exists(self.COCO_MODEL_PATH):
            utils.download_trained_weights(self.COCO_MODEL_PATH)
        self.config = InferenceConfig()
        self.config.display()
        self.model = modellib.MaskRCNN(mode="inference", model_dir=self.MODEL_DIR, config=self.config)
        self.model.load_weights(self.COCO_MODEL_PATH, by_name=True)

    def get_mask(self,classes, masks, ids):
        true_mask = np.full_like(masks[:, :, 0], 0)
        for i, c in enumerate(ids):
            if c in classes:
                true_mask += masks[:, :, i]
        true_mask = true_mask > 0
        return true_mask

    def classmask(self, image, classes):
        results = self.model.detect([image], verbose=1)

        # Visualize results
        r = results[0]

        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    class_names, r['scores'])
        print(type(image))
        mask = self.get_mask(classes, r['masks'], r['class_ids'])
        for c in range(3):
            image[:, :, c] = np.where(mask == 1,
                                      image[:, :, c],
                                      0)

        img = Image.fromarray(image, 'RGB')
        img.save('my.png')
        img.show()


# Create model object in inference mode.


# Load weights trained on MS-COCO

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')


# Load a random image from the images folder

# Run detection



