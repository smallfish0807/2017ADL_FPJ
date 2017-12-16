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
flags.DEFINE_integer("max_epoch", 100000, "# of step in an epoch")
flags.DEFINE_integer("test_step", 1, "# of step to test a model")
flags.DEFINE_integer("save_step", 1, "# of step to save a model")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("grad_clip", 1, "value of gradient to be used for clipping")
flags.DEFINE_boolean("use_gpu", True, "whether to use gpu for training")

# data
flags.DEFINE_string("data", "mnist", "name of dataset [mnist, cifar, imageNet]")
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
    preprocess_conf(conf)

    DATA_DIR = os.path.join(conf.data_dir, conf.data)
    SAMPLE_DIR = os.path.join(conf.sample_dir, conf.data, model_dir)

    print('model_dir:', model_dir)
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

        train_step_per_epoch = int(mnist.train.num_examples / conf.batch_size)
        test_step_per_epoch = int(mnist.test.num_examples / conf.batch_size)
    elif conf.data == "cifar":
        from cifar10 import IMAGE_SIZE, inputs

        maybe_download_and_extract(DATA_DIR)
        images, labels = inputs(eval_data=False,
                data_dir=os.path.join(DATA_DIR, 'cifar-10-batches-bin'), batch_size=conf.batch_size)

        height, width, channel = IMAGE_SIZE, IMAGE_SIZE, 3
    elif conf.data == 'imageNet':
        #images_all = get_all_imageNet_images('./data/valid_32x32/')
        #np.save('./npy/images_all_valid', images_all)
        images_all = np.load('./npy/images_all_valid.npy')
        np.random.shuffle(images_all)

        images_all_train = images_all[1000:]
        images_all_test = images_all[:1000]
        height, width, channel = 32, 32, 1
        train_step_per_epoch = int(images_all_train.shape[0] / conf.batch_size)
        test_step_per_epoch = int(images_all_test.shape[0] / conf.batch_size)


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        network = Network(sess, conf, height, width, channel)

        stat = Statistic(sess, conf.data, model_dir, tf.trainable_variables(), conf.test_step)
        stat.load_model()

        if conf.is_train:
            logger.info("Training starts!")

            initial_step = stat.get_t() if stat else 0

            for epoch in range(initial_step, conf.max_epoch):
                # 1. train
                total_train_costs = []
                for idx in range(train_step_per_epoch):
                    if conf.data == "mnist":
                        #next_train_batch(conf.batch_size).shape: (100, 784)
                        images = binarize(next_train_batch(conf.batch_size)) \
                            .reshape([conf.batch_size, height, width, channel])
                    elif conf.data == 'imageNet':
                        #images = binarize(get_batch(images_all_train, conf.batch_size, idx))
                        images = get_batch(images_all_train, conf.batch_size, idx)


                    cost = network.test(images, with_update=True)
                    total_train_costs.append(cost)

                #logger.info('epoch: '+str(epoch)+' => '+' train l: '+str(np.mean(total_train_costs)))
                # 2. test
                if epoch % conf.test_step == 0:
                    total_test_costs = []
                    for idx in range(test_step_per_epoch):
                        if conf.data == "mnist":
                            images = binarize(next_test_batch(conf.batch_size)) \
                                .reshape([conf.batch_size, height, width, channel])
                        elif conf.data == 'imageNet':
                            #images = binarize(get_batch(images_all_test, conf.batch_size, idx))
                            images = get_batch(images_all_test, conf.batch_size, idx)

                        cost = network.test(images, with_update=False)
                        total_test_costs.append(cost)

                        # save images generated by test
                        if epoch == 1 and idx == 0:
                            save_images(images, height, width, 10, 10, directory=SAMPLE_DIR, prefix="test_ori")
                        if epoch % 10 == 0 and idx == 0:
                            test_out = network.predict(images)
                            save_images(test_out, height, width, 10, 10, directory=SAMPLE_DIR, prefix="test_out_epoch_%s" % epoch)
                            test_out_half = network.test_generate_half(images)
                            save_images(test_out_half, height, width, 10, 10, directory=SAMPLE_DIR, prefix="test_out_half_epoch_%s" % epoch)

                    #print('total_train_costs=', total_train_costs, ' total_test_costs=', total_test_costs)
                    avg_train_cost, avg_test_cost = np.mean(total_train_costs), np.mean(total_test_costs)
                    logger.info('epoch: '+str(epoch)+' => '+' train l: '+str(avg_train_cost)+' test l: '+str(avg_test_cost))

                    # stat and save model
                    #if epoch % conf.save_step == 0:
                        #stat.on_step(avg_train_cost, avg_test_cost)

                # 3. generate samples
                #samples = network.generate()
                #save_images(samples, height, width, 10, 10, directory=SAMPLE_DIR, prefix="epoch_%s" % epoch)

        else:
            logger.info("Image generation starts!")

            samples = network.generate()
            save_images(samples, height, width, 10, 10, directory=SAMPLE_DIR)


if __name__ == "__main__":
    tf.app.run()
