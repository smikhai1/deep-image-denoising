import os
import cv2
import skimage
from skimage.filters import median
from skimage.morphology import rectangle
import numpy as np

def preprocess(kernel_size=7):

    """
    Filter all images in the specified folder using median filter

    :param kernel_size: size of squared kernel used in median filter
    :return:
    """

    # path to original data
    PATH_ORIG =  '/Users/mikhail/projects/edu/research/denoising/data/origin/noisy/'
    PATH_CLEAN = '/Users/mikhail/projects/edu/research/denoising/data/origin/clean/'

    images_names = os.listdir(PATH_ORIG)

    for i, image_name in enumerate(images_names):
        #img = cv2.imread(PATH_ORIG + image_name)
        # img_clean = cv2.medianBlur(img, 7)
        # cv2.imwrite(PATH_CLEAN + new_name, img_clean)


        img = skimage.io.imread(PATH_ORIG + image_name, as_gray=True)
        # filter image using median filter

        img_clean = median(img, selem=rectangle(kernel_size, kernel_size))

        # save image in another folder with different name
        new_name = 'F' + image_name[1:]
        skimage.io.imsave(PATH_CLEAN + new_name, img_clean)

        if (i % 200 == 0) and (i != 0):
            print(f'{i} images have been filtered')