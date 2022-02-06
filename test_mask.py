import os

import skimage
import random

from gen_chair import coco
from gen_chair import segmentation
from PIL import Image



IMAGE_DIR = "/home/victor/Desktop/img2img/data/sitting_in_chair/"
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

if __name__ == "__main__":
    chair = 57
    person = 1
    print(IMAGE_DIR)
    file_names = next(os.walk(IMAGE_DIR))[2]
    image = skimage.io.imread(os.path.join(IMAGE_DIR, random.choice(file_names)))
    personimg = image.copy()
    pcimg = image.copy()
    seg = segmentation(InferenceConfig)
    seg.classmask(pcimg,[person, chair])
    seg.classmask(personimg, [person])
    img = Image.fromarray(image, 'RGB')
    img.save('my1.png')
    img.show()
    img = Image.fromarray(personimg, 'RGB')
    img.save('my2.png')
    img.show()
    img = Image.fromarray(pcimg, 'RGB')
    img.save('my3.png')
    img.show()