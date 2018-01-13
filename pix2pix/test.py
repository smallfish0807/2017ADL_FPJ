import numpy as np
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util import util

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
# test

# Create or clear dir for saving generated samples
if os.path.exists(opt.testing_path):
    shutil.rmtree(opt.testing_path)

os.makedirs(opt.testing_path)
img_comb = {}
img_comb_row = {}
for j,data in enumerate(dataset):
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    # Combine 100 image into one
    for label, image_numpy in visuals.items():
        if (j+1) % 10 == 1:
            img_comb_row[label] = image_numpy
        else:
            img_comb_row[label] = np.concatenate([img_comb_row[label], image_numpy], 1)

        if j == 9:
            img_comb[label] = img_comb_row[label]
        elif (j+1) % 10 == 0:
            img_comb[label] = np.concatenate([img_comb[label], img_comb_row[label]], 0)

# Save
for label, image_numpy in img_comb.items():
    image_name = '%s_%s.png' % (opt.which_epoch, label)
    save_path = os.path.join(opt.testing_path, image_name)
    util.save_image(image_numpy, save_path)
