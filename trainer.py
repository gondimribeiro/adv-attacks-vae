import matplotlib
matplotlib.use("Agg")

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import seaborn as sns
import tensorflow as tf
import sys

import celeba_utils
import mnist_utils
import svhn_utils

from draw import DRAW
from cvae import ConvVAE
from vae import VAE


parser = argparse.ArgumentParser()

# General arguments
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)

parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--dataset', type=str, default='celeba')
parser.add_argument('--architecture', type=str, default='draw')
parser.add_argument('--dir', type=str, default='./celeba/draw')
parser.add_argument('--adv_learning_rate', type=float, default=1e-3)


parser.add_argument('--is_gaussian', dest='is_gaussian', action='store_true')
parser.set_defaults(is_gaussian=False)


parser.add_argument('--random_sampling',
                    dest='random_sampling', action='store_true')
parser.set_defaults(random_sampling=False)

# Number of latent variables
parser.add_argument('--z_size', type=int, default=32)

# DRAW arguments
parser.add_argument('--T', type=int, default=8)
parser.add_argument('--enc_size', type=int, default=256)
parser.add_argument('--dec_size', type=int, default=256)
parser.add_argument('--read_n', type=int, default=12)
parser.add_argument('--write_n', type=int, default=12)

parser.add_argument('--read_attn', dest='read_attn', action='store_true')
parser.set_defaults(read_attn=False)

parser.add_argument('--write_attn', dest='write_attn', action='store_true')
parser.set_defaults(write_attn=False)


# ConvVAE arguments
parser.add_argument('--dense_size', type=int, default=512)

# Parser
args, unknown = parser.parse_known_args()

if len(unknown) > 0:
    print("Invalid arguments %s" % unknown)
    parser.print_help()
    sys.exit()

args = vars(args)

print('Config')
for key in args:
    print(key, ':', args[key])
print("----------------------------------------")

# LOAD DATA
np.random.seed(0)
tf.set_random_seed(0)
sns.set()

if args['dataset'] == 'celeba':
    data_utils = celeba_utils
    train_x, val_x, test_x = celeba_utils.load_data(args['is_gaussian'])
    print("CelebA loaded")
elif args['dataset'] == 'svhn':
    data_utils = svhn_utils
    train_x, _, val_x, _, test_x, _ = svhn_utils.load_data(args['is_gaussian'])
    print("SVHN loaded")
elif args['dataset'] == 'mnist':
    data_utils = mnist_utils
    train_x, _, val_x, _, test_x, _ = mnist_utils.load_data(
        args['is_gaussian'])
else:
    sys.exit("No dataset %s" % args['dataset'])

print("Training samples %d" % train_x.shape[0])
print("Validation samples %d" % val_x.shape[0])
print("Test samples %d" % test_x.shape[0])
print("----------------------------------------")


# Build graph
config = dict()
config['architecture'] = args['architecture']
config['dataset'] = args['dataset']
config['is_gaussian'] = args['is_gaussian']
config['batch_size'] = args['batch_size']
config['num_epochs'] = args['epochs']
config['learning_rate'] = args['learning_rate']
config['adv_learning_rate'] = args['adv_learning_rate']
config['input_shape'] = [data_utils.img_dim,  data_utils.img_dim,
                         data_utils.n_chan]
config['data_mean'] = data_utils.data_mean
config['data_std'] = data_utils.data_std
config['data_max_pixel'] = data_utils.max_pixel
config['is_attacking'] = False

tf.reset_default_graph()
if args['architecture'] == 'draw':
    config['enc_size'] = args['enc_size']
    config['dec_size'] = args['dec_size']
    config['read_n'] = args['read_n']
    config['write_n'] = args['write_n']
    config['read_attn'] = args['read_attn']
    config['write_attn'] = args['write_attn']
    config['n_z'] = args['z_size']
    config['T'] = args['T']
    config['z_size'] = args['z_size'] * args['T']

    vae = DRAW(config)
    print("----------------------------------------")
    print("DRAW graph loaded")

elif args['architecture'] == 'cvae':
    conv_layers = []
    deconv_layers = []
    if args['dataset'] == 'svhn':
        conv_layers.append(dict(n_filters=32, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=64, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=128, filter_size=4, stride=2))

        gen_init_shape = [4, 4, int(args['dense_size'] / 16)]

        deconv_layers.append(
            dict(n_filters=128, filter_size=5, stride=2, pad='same'))
        deconv_layers.append(
            dict(n_filters=64, filter_size=5, stride=2, pad='same'))
        deconv_layers.append(
            dict(n_filters=32, filter_size=5, stride=2, pad='same'))

    elif args['dataset'] == 'celeba':
        conv_layers.append(dict(n_filters=32, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=64, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=128, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=256, filter_size=4, stride=2))

        gen_init_shape = [4, 4, int(args['dense_size'] / 16)]

        deconv_layers.append(
            dict(n_filters=256, filter_size=5, stride=2, pad='same'))
        deconv_layers.append(
            dict(n_filters=128, filter_size=5, stride=2, pad='same'))
        deconv_layers.append(
            dict(n_filters=64, filter_size=5, stride=2, pad='same'))
        deconv_layers.append(
            dict(n_filters=32, filter_size=5, stride=2, pad='same'))

    elif args['dataset'] == 'mnist':
        conv_layers.append(dict(n_filters=32, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=64, filter_size=4, stride=2))
        conv_layers.append(dict(n_filters=128, filter_size=4, stride=2))

        gen_init_shape = [1, 1, args['dense_size']]

        deconv_layers.append(
            dict(n_filters=128, filter_size=3, stride=2, pad='valid'))
        deconv_layers.append(
            dict(n_filters=64, filter_size=3, stride=2, pad='valid'))
        deconv_layers.append(
            dict(n_filters=32, filter_size=2, stride=2, pad='valid'))
        deconv_layers.append(
            dict(n_filters=16, filter_size=2, stride=2, pad='valid'))

    config['conv_layers'] = conv_layers
    config['deconv_layers'] = deconv_layers
    config['dense_layers'] = [args['dense_size']]
    config['gen_dense_layers'] = [args['dense_size']]
    config['gen_init_shape'] = gen_init_shape
    config['transfer_fct'] = tf.nn.elu
    config['output'] = dict(filter_size=4, stride=1)
    config['z_size'] = args['z_size']

    vae = ConvVAE(config)
    print("----------------------------------------")
    print("ConvVAE graph loaded")

elif args['architecture'] == 'vae':
    enc_layers = [512, 512]
    dec_layers = [512, 512]

    config['enc_layers'] = enc_layers
    config['dec_layers'] = dec_layers
    config['transfer_fct'] = tf.nn.softplus
    config['z_size'] = args['z_size']

    vae = VAE(config)
    print("----------------------------------------")
    print("VAE graph loaded")

else:
    sys.exit("No architecture %s" % args['architecture'])

tf.get_default_graph().finalize()

# Test images
img_idx = np.random.choice(test_x.shape[0], 10)
test_batch = test_x[img_idx]

model_path = args['dir']

if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(model_path + '/model'):
    os.makedirs(model_path + '/model')
if not os.path.exists(model_path + '/results'):
    os.makedirs(model_path + '/results')
with open(model_path + '/architecture.txt', 'w') as fp:
    for key in config:
        fp.write(
                ('%s : %s\n' % (key, config[key]))
        )
with open(model_path + '/architecture.pkl', 'wb') as fp:
    pickle.dump(config, fp)

total_parameters = 0
for variable in tf.trainable_variables():
    shape = variable.get_shape()
    variable_parameters = 1
    print(variable)
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters
print('# of parameters = %d' % total_parameters)
print("----------------------------------------")


print('Training...')
data_utils.reconstruct(
    vae,
    test_batch,
    model_path + '/results/rec.epoch=0.png',
    normalize=args['is_gaussian']
)

loss, val_loss = vae.fit(
    train_x,
    val_x,
    args['epochs'],
    model_path + '/model',
    img_path=model_path + '/results',
    X_test=test_batch,
    f_reconstruct=data_utils.reconstruct,
    random_sampling=args['random_sampling']
)

plt.figure()
plt.plot(range(len(loss)), loss, label='training')
plt.plot(range(len(val_loss)), val_loss, label='validation')
plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig(model_path + '/results/training.png')

vae.load(tf.train.latest_checkpoint(model_path + '/model/'))

data_utils.reconstruct(
    vae,
    test_batch,
    model_path + '/results/rec.epoch=%d.png' % args['epochs'],
    normalize=args['is_gaussian'],

)

vae.close()
