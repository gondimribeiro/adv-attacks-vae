'''
Dataset from: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
'''

import matplotlib
matplotlib.use("Agg")

import math
import matplotlib.pyplot as plt
import numpy as np


data_mean = 0.431751299266
data_std = 0.300219581459
img_dim = 64
n_chan = 3
max_pixel = 1.


def show(img, normalize, i, title="", num_rows=-3, path=None):
    img = img.copy().reshape(img_dim, img_dim, 3)

    if normalize:
        img *= data_std
        img += data_mean
        img = np.clip(img, 0., 1.)

    if num_rows < 0:
        plt.subplot(2, -num_rows, i)
    else:
        plt.subplot(num_rows, 2, i)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")


def load_data(normalize=True):
    # LOAD DATA
    if normalize:
        train_x = np.load('./data/celeba/celeba_64x64_train_normalized.npy',
                          mmap_mode='r')
        test_x = np.load('./data/celeba/celeba_64x64_test_normalized.npy',
                         mmap_mode='r')
    else:
        train_x = np.load('./data/celeba/celeba_64x64_train.npy',
                          mmap_mode='r')
        test_x = np.load('./data/celeba/celeba_64x64_test.npy',
                         mmap_mode='r')

    val_idx = math.ceil(train_x.shape[0] * 0.1)
    val_x = train_x[:val_idx]

    train_x = train_x[val_idx:]

    return train_x, val_x, test_x


def reconstruct(vae, batch, path=None, normalize=True):
    plt.figure(figsize=(8, 30))
    num_imgs = batch.shape[0]
    batch_rec = vae.batch_reconstruct(batch)
    for i in range(num_imgs):
        show(batch[i], normalize, 2 * i + 1, 'original', num_imgs)
        show(batch_rec[i], normalize, 2 * i + 2, 'reconstr', num_imgs)

    if path is None:
        plt.show()
    else:
        plt.savefig(path)
    plt.close()


def input(img, batch_size):
    return np.tile(img, (batch_size, 1, 1, 1)).reshape(batch_size,
                                                       img_dim, img_dim, 3)


def dist(imgs, img, batch_size):
    if imgs.shape == (batch_size, img_dim, img_dim, 3):
        imgs = imgs.reshape(batch_size, 12288)

    assert imgs.shape == (batch_size, 12288)

    img = img.reshape((12288,))

    imgs_pixels = imgs * data_std + data_mean
    img_pixels = img * data_std + data_mean

    diff = np.linalg.norm(imgs_pixels - img_pixels, axis=1)
    return np.mean(diff), np.std(diff)
