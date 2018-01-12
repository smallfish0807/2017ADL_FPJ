import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import ntpath
import os
from util import util
import shutil
import numpy as np
#from util.visualizer import Visualizer

opt = TrainOptions().parse()


# Load dataset_train
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# Load dataset_test
opt.batchSize = 1
opt.phase = 'test'
data_loader_test = CreateDataLoader(opt)
dataset_test = data_loader_test.load_data()
dataset_size = len(data_loader_test)
print('#testing images = %d' % dataset_size)

# Create or clear dir for saving generated samples
if os.path.exists(opt.testing_path):
    shutil.rmtree(opt.testing_path)
os.makedirs(opt.testing_path)

# Model
model = create_model(opt)
#visualizer = Visualizer(opt)
total_steps = 0

# Start training
print('Start training')
for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    error_sum_epoch = {'G_GAN':0, 'G_L1':0, 'D_real':0, 'D_fake':0}
    error_sum = {'G_GAN':0, 'G_L1':0, 'D_real':0, 'D_fake':0}

    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        #visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
        errors = model.get_current_errors()
        for k,v in errors.items():
            error_sum_epoch[k] += v
            error_sum[k] += v

        if total_steps % opt.print_freq == 0:
            for k,v in errors.items():
                error_sum[k] /= opt.print_freq
            print('Ep %d Step %d:' % (epoch, i), error_sum)
            error_sum = {'G_GAN':0, 'G_L1':0, 'D_real':0, 'D_fake':0}

        '''
        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            print(epoch, epoch_iter, errors, t)
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')
        '''

    # Save model and generate test samples
    if epoch % opt.save_epoch_freq == 0:
        ### Saving model ###
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save(epoch)

        ### Testing ###
        img_comb = {}
        img_comb_row = {}
        for j,data in enumerate(dataset_test):
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
            image_name = '%s_%s.png' % (epoch, label)
            save_path = os.path.join(opt.testing_path, image_name)
            util.save_image(image_numpy, save_path)

    for k,v in errors.items():
        error_sum_epoch[k] /= len(data_loader)
    print('Epoch %d/%d ' % (epoch, opt.niter + opt.niter_decay), error_sum_epoch)
    # update parameters
    model.update_learning_rate()
