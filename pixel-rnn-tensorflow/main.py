import os
import logging
logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S")

import numpy as np
from tqdm import trange
import tensorflow as tf

from utils import *
from network import Network
from statistic import Statistic
from imageNet import *

flags = tf.app.flags

# network
flags.DEFINE_string("model", "pixel_cnn", "name of model [pixel_rnn, pixel_cnn]")
flags.DEFINE_integer("batch_size", 100, "size of a batch")
flags.DEFINE_integer("hidden_dims", 16, "dimesion of hidden states of LSTM or Conv layers")
flags.DEFINE_integer("recurrent_length", 7, "the length of LSTM or Conv layers")
flags.DEFINE_integer("out_hidden_dims", 32, "dimesion of hidden states of output Conv layers")
flags.DEFINE_integer("out_recurrent_length", 2, "the length of output Conv layers")
flags.DEFINE_boolean("use_residual", False, "whether to use residual connections or not")
# flags.DEFINE_boolean("use_dynamic_rnn", False, "whether to use dynamic_rnn or not")

# training
#flags.DEFINE_integer("max_epoch", 100000, "# of step in an epoch")
flags.DEFINE_integer("max_epoch", 3, "# of step in an epoch")
flags.DEFINE_integer("test_step", 100, "# of step to test a model")
flags.DEFINE_integer("save_step", 1000, "# of step to save a model")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")

# data
flags.DEFINE_string("data", "mnist", "name of dataset [mnist, cifar]")
flags.DEFINE_string("data_dir", "data", "name of data directory")
flags.DEFINE_string("sample_dir", "samples", "name of sample directory")

# Debug
flags.DEFINE_boolean("is_train", True, "training or testing")
flags.DEFINE_boolean("display", False, "whether to display the training results or not")
flags.DEFINE_string("log_level", "INFO", "log level [DEBUG, INFO, WARNING, ERROR, CRITICAL]")
flags.DEFINE_integer("random_seed", 123, "random seed for python")

conf = flags.FLAGS

# logging
logger = logging.getLogger()
logger.setLevel(conf.log_level)

# random seed
tf.set_random_seed(conf.random_seed)
np.random.seed(conf.random_seed)

def main(_):
    model_dir = get_model_dir(conf,
            ['data_dir', 'sample_dir', 'max_epoch', 'test_step', 'save_step',
             'is_train', 'random_seed', 'log_level', 'display'])
    print('model_dir:', model_dir)
    preprocess_conf(conf)

    DATA_DIR = os.path.join(conf.data_dir, conf.data)
    SAMPLE_DIR = os.path.join(conf.sample_dir, conf.data, model_dir)
    print('DATA_DIR:', DATA_DIR)
    print('SAMPLE_DIR:', SAMPLE_DIR)

    check_and_create_dir(DATA_DIR)
    check_and_create_dir(SAMPLE_DIR)

    # 0. prepare datasets
    if conf.data == "mnist":
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

        next_train_batch = lambda x: mnist.train.next_batch(x)[0]
        next_test_batch = lambda x: mnist.test.next_batch(x)[0]

        height, width, channel = 28, 28, 1

        train_step_per_epoch = mnist.train.num_examples / conf.batch_size
        test_step_per_epoch = mnist.test.num_examples / conf.batch_size
    elif conf.data == "cifar":
        from cifar10 import IMAGE_SIZE, inputs

        maybe_download_and_extract(DATA_DIR)
        images, labels = inputs(eval_data=False,
                data_dir=os.path.join(DATA_DIR, 'cifar-10-batches-bin'), batch_size=conf.batch_size)

        height, width, channel = IMAGE_SIZE, IMAGE_SIZE, 3
    elif conf.data == 'imageNet':
        #images_all_train = get_all_imageNet_images('../data_small/train_32x32/')
        #images_all_valid = get_all_imageNet_images('../data_small/valid_32x32/')
        #np.save('./npy/images_all_train_small', images_all_train)
        #np.save('./npy/images_all_valid_small', images_all_valid)
        images_all_train = np.load('./npy/images_all_train_small.npy')
        images_all_valid = np.load('./npy/images_all_valid_small.npy')
        height, width, channel = 32, 32, 3
        train_step_per_epoch = images_all_train.shape[0] / conf.batch_size
        valid_step_per_epoch = images_all_valid.shape[0] / conf.batch_size


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        network = Network(sess, conf, height, width, channel)

        stat = Statistic(sess, conf.data, model_dir, tf.trainable_variables(), conf.test_step)
        stat.load_model()

        if conf.is_train:
            logger.info("Training starts!")

            initial_step = stat.get_t() if stat else 0
            iterator = trange(conf.max_epoch, ncols=70, initial=initial_step)

            for epoch in iterator:
                # 1. train
                total_train_costs = []
                for idx in xrange(train_step_per_epoch):
                    if conf.data == "mnist":
                        #next_train_batch(conf.batch_size).shape: (100, 784)
                        images = binarize(next_train_batch(conf.batch_size)) \
                            .reshape([conf.batch_size, height, width, channel])
                    elif conf.data == 'imageNet':
                        images = get_batch(images_all_train, conf.batch_size, epoch)


                    cost = network.test(images, with_update=True)
                    total_train_costs.append(cost)

                # 2. test
                total_test_costs = []
                for idx in xrange(test_step_per_epoch):
                    if conf.data == "mnist":
                        images = binarize(next_test_batch(conf.batch_size)) \
                            .reshape([conf.batch_size, height, width, channel])
                    elif conf.data == 'imageNet':
                        images = get_batch(images_all_test, conf.batch_size, epoch)

                    cost = network.test(images, with_update=False)
                    total_test_costs.append(cost)

                avg_train_cost, avg_test_cost = np.mean(total_train_costs), np.mean(total_test_costs)

                stat.on_step(avg_train_cost, avg_test_cost)

                # 3. generate samples
                samples = network.generate()
                save_images(samples, height, width, 10, 10,
                        directory=SAMPLE_DIR, prefix="epoch_%s" % epoch)

                iterator.set_description("train l: %.3f, test l: %.3f" % (avg_train_cost, avg_test_cost))
                print
        else:
            logger.info("Image generation starts!")

            samples = network.generate()
            save_images(samples, height, width, 10, 10, directory=SAMPLE_DIR)


if __name__ == "__main__":
    tf.app.run()
