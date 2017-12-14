import numpy as np
from PIL import Image

from os import listdir
from os.path import isfile, join

def preprosess_img(x):
    #TODO
    ##
    #x[..., 0] -= 103.939
    #x[..., 1] -= 116.779
    #x[..., 2] -= 123.68
    ##
    #x /= 127.5
    #x -= 1.
    ##
    #x /= 255
    ##
    return x

def get_all_imageNet_images(file_path):
    filename_list = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    images_all = []
    for fn in filename_list:
        I = np.asarray(Image.open(join(file_path, fn)), dtype=np.float32)
        images_all.append(I)
    return np.asarray(images_all)

def get_batch(images_all, batch_size, i):
    return images_all[i*batch_size+1: (i+1)*batch_size]
