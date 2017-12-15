import numpy as np
from PIL import Image
import scipy

from os import listdir
from os.path import isfile, join

def prepro(o):
    """
    Call this function to preprocess RGB image to grayscale image
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, o.shape[:-1])
    resized = resized / 255
    return np.expand_dims(resized.astype(np.float32),axis=2)

def get_all_imageNet_images(file_path):
    filename_list = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    images_all = []
    for fn in filename_list:
        I = np.asarray(Image.open(join(file_path, fn)), dtype=np.float32)
        images_all.append(prepro(I))
    return np.asarray(images_all)

def get_batch(images_all, batch_size, i):
    return images_all[i*batch_size+1: (i+1)*batch_size]
