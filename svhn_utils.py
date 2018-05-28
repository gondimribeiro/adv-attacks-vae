import matplotlib
matplotlib.use("Agg")

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


data_mean = 115.11177966923525
data_std = 50.819267906232888
img_dim = 32
max_pixel = 255.
n_chan = 3


def show(img, normalize, i, title="", num_rows=-3, path=None):
    img = img.copy().reshape(img_dim, img_dim, 3)

    if normalize:
        img *= data_std
        img += data_mean
        img /= 255.
        img = np.clip(img, 0., 1.)

    if num_rows < 0:
        plt.subplot(2, -num_rows, i)
    else:
        plt.subplot(num_rows, 2, i)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


def reconstruct(cvae, batch, path=None, normalize=True):
    plt.figure(figsize=(8, 30))
    num_imgs = batch.shape[0]
    batch_rec = cvae.batch_reconstruct(batch)
    for i in range(num_imgs):
        show(batch[i], normalize, 2 * i + 1, 'original', num_imgs)
        show(batch_rec[i], normalize, 2 * i + 2, 'reconstr', num_imgs)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


def load_data(normalize=True):
    # LOAD DATA
    svhn_train = loadmat('./data/svhn/train_32x32.mat')
    svhn_test = loadmat('./data/svhn/test_32x32.mat')

    train_x = np.rollaxis(svhn_train['X'], 3).astype(np.float32)
    test_x = np.rollaxis(svhn_test['X'], 3).astype(np.float32)

    train_y = svhn_train['y'].flatten() - 1
    test_y = svhn_test['y'].flatten() - 1

    idx = np.random.permutation(train_x.shape[0])
    train_x = train_x[idx]
    train_y = train_y[idx]

    if normalize:
        train_x = (train_x - data_mean) / data_std
        test_x = (test_x - data_mean) / data_std
    else:
        train_x = train_x / 255.
        test_x = test_x / 255.

    val_idx = math.ceil(train_x.shape[0] * 0.1)
    val_x = train_x[:val_idx]
    val_y = train_y[:val_idx]

    train_x = train_x[val_idx:]
    train_y = train_y[val_idx:]

    return train_x, train_y, val_x, val_y, test_x, test_y


def input(img, batch_size):
    return np.tile(img, (batch_size, 1, 1, 1)).reshape(batch_size,
                                                       img_dim, img_dim, 3)


def dist(imgs, img, batch_size):
    if imgs.shape == (batch_size, 32, 32, 3):
        imgs = imgs.reshape(batch_size, 3072)

    assert imgs.shape == (batch_size, 3072)

    img = img.reshape((3072,))

    imgs_pixels = imgs * data_std + data_mean
    img_pixels = img * data_std + data_mean

    diff = np.linalg.norm(imgs_pixels - img_pixels, axis=1)
    return np.mean(diff), np.std(diff)
