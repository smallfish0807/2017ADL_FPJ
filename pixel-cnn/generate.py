"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr-gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os
import sys
import time
import json
import argparse

import numpy as np
import tensorflow as tf

import pixel_cnn_pp.nn as nn
import pixel_cnn_pp.plotting as plotting
from pixel_cnn_pp.model import model_spec
import data.cifar10_data as cifar10_data
import data.imagenet_data as imagenet_data

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str,
                    default='../data', help='Location for the dataset')
parser.add_argument('-o', '--save_dir', type=str, default='../save',
                    help='Location for parameter checkpoints and samples')
parser.add_argument('-d', '--data_set', type=str,
                    default='imagenet', help='Can be either cifar|imagenet')
parser.add_argument('-t', '--save_interval', type=int, default=1,
                    help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-r', '--load_params', type=str, default=None,
                    help='Restore training from which model checkpoint?')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                    help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160,
                    help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                    help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu',
                    help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional',
                    action='store_true', help='Condition generative model on labels?')
# optimization
parser.add_argument('-l', '--learning_rate', type=float,
                    default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                    help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=12,
                    help='Batch size during training per GPU')
parser.add_argument('-a', '--init_batch_size', type=int, default=100,
                    help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5,
                    help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int,
                    default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=1,
                    help='How many GPUs to distribute the training across?')
parser.add_argument('--gpu_memory', type=float, default=0.5, help='Used percentage of GPU memory.')

# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995,
                    help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1,
                    help='Random seed to use')
# visialization
parser.add_argument('--num_sample', type=int, default=100)
parser.add_argument('--num_gen', type=int, default=1)

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4,
                                  separators=(',', ':')))  # pretty print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
DataLoader = {'cifar': cifar10_data.DataLoader,
              'imagenet': imagenet_data.DataLoader}[args.data_set]
train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu,
                        rng=rng, shuffle=True, return_labels=args.class_conditional)
test_data = DataLoader(args.data_dir, 'test', args.batch_size *
                       args.nr_gpu, shuffle=False, return_labels=args.class_conditional)
obs_shape = test_data.get_observation_size()  # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)
      for i in range(args.nr_gpu)]

h_init = None
h_sample = [None] * args.nr_gpu
hs = h_sample

print('Creating model')
# create the model
model_opt = {'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters,
             'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity}
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
gen_par_init = model(x_init, h_init, init=True,
                dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))

# get loss gradients over multiple GPUs
grads = []
loss_gen = []
loss_gen_test = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        gen_par = model(xs[i], hs[i], ema=None,
                        dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))
        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params))
        # test
        gen_par = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(nn.discretized_mix_logistic_loss(xs[i], gen_par))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(
        all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[
    0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)
bits_per_dim_test = loss_gen_test[
    0] / (args.nr_gpu * np.log(2.) * np.prod(obs_shape) * args.batch_size)

# sample from the model
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        gen_par = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        new_x_gen.append(nn.sample_from_discretized_mix_logistic(
            gen_par, args.nr_logistic_mix))


def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
             for i in range(args.nr_gpu)]
    #print('ori min =', np.min(x_gen), 'max =', np.max(x_gen), 'type =', x_gen[0].dtype)
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(
                new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    #print('ori min =', np.min(x_gen), 'max =', np.max(x_gen), 'type =', x_gen[0].dtype)
    return np.concatenate(x_gen, axis=0)

def sample_half(sess):
    #x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
    #         for i in range(args.nr_gpu)]
    begin = time.time()
    test_data.reset()
    x_gen, test_img = [], []
    for n in range(args.num_sample//(args.nr_gpu*args.batch_size) + 1):
        for i in range(args.nr_gpu):
            x = np.copy(test_data.next(args.batch_size)).astype(np.float32)
            test_img.append(np.copy(x))
            x =  (x / (255. / 2)) - 1.
            x[:, int(obs_shape[0]//2+1):, :, :] = -1.
            x_gen.append(x)
    test_data.reset()
    x_gen_input = np.concatenate(x_gen, axis=0)
    test_img = np.concatenate(test_img, axis=0)

    for n in range(int(len(x_gen)/args.nr_gpu)):
        for yi in range(obs_shape[0]//2+1, obs_shape[0]):
        #for yi in range(obs_shape[0]):
            for xi in range(obs_shape[1]):
                new_x_gen_np = sess.run(new_x_gen,
                    {xs[i]: x_gen[i+n*args.nr_gpu] for i in range(args.nr_gpu)})
                for i in range(args.nr_gpu):
                    x_gen[i+n*args.nr_gpu][:, yi, xi, :] = new_x_gen_np[i][:, yi, xi, :]
    x_gen = np.concatenate(x_gen, axis=0)
    print('sample %d images: %d s' % (len(x_gen), time.time()-begin))
    return x_gen, x_gen_input, test_img


# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=10000)

# turn numpy inputs into feed_dict for use with tensorflow


def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x, y = data
    else:
        x = data
        y = None
    # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    x = np.cast[np.float32]((x - 127.5) / 127.5)
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
print('starting training')
lr = args.learning_rate

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # init
    # manually retrieve exactly init_batch_size examples
    feed_dict = make_feed_dict(
        train_data.next(args.init_batch_size), init=True)
    train_data.reset()  # rewind the iterator back to 0 to do one full epoch
    #sess.run(initializer, feed_dict)
    sess.run(initializer)
    sess.run(gen_par_init, feed_dict)
    print('initializing the model...')
    if args.load_params is not None:
        #ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt' + str(args.load_params)
        ckpt_file = args.load_params
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    # generate samples from the model

    def plot_an_img(img, filename):
        img_tile = plotting.img_tile(img[:args.num_sample],
            tile_shape=(int(np.sqrt(args.num_sample)), int(np.sqrt(args.num_sample))),
            aspect_ratio=1.0, border_color=1.0, stretch=True)
        img = plotting.plot_img(img_tile)
        plotting.plt.savefig(os.path.join(
            args.save_dir, filename))
        plotting.plt.close('all')

    #sample_x = sample_from_model(sess)
    for i in range(args.num_gen):
        print('Generateing samples', i)
        sample_x, sample_input, test_img = sample_half(sess)
        #img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(
        #    args.batch_size * args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)

        plot_an_img(sample_x, '%s_sample_%d.png' % (args.data_set, i))
        if i == 0:
            plot_an_img(sample_input, '%s_sample_input.png' % (args.data_set))
            plot_an_img(test_img, '%s_test_img.png' % (args.data_set))

    '''
    sample_x = sample_from_model(sess)
    img_tile = plotting.img_tile(sample_x[:int(np.floor(np.sqrt(
        args.batch_size * args.nr_gpu))**2)], aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile)
    plotting.plt.savefig(os.path.join(
        args.save_dir, '%s_sample_ori.png' % (args.data_set)))
    plotting.plt.close('all')
    '''
