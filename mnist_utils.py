import matplotlib
matplotlib.use("Agg")

import math
import matplotlib.pyplot as plt
import numpy as np
from mnist import MNIST


# TODO: clean up

img_dim = 28
n_chan = 1
max_pixel = 1.
data_mean = 0
data_std = 1.


def show(img, normalize, i, title="", num_rows=-3, path=None):
    img = img.copy().reshape(img_dim, img_dim)

    if num_rows < 0:
        plt.subplot(2, -num_rows, i)
    else:
        plt.subplot(num_rows, 2, i)

    plt.imshow(img, vmin=0, vmax=1, cmap="gray")
    plt.title(title)
    plt.axis("off")


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


def load_data(normalize=True):
    # LOAD DATA
    mndata = MNIST("./data/mnist")

    mndata.gz = True

    train_x, train_y = mndata.load_training()
    test_x, test_y = mndata.load_testing()

    train_x = np.array(train_x, dtype=np.float32).reshape(
        len(train_x), 28, 28, 1) / 255.0
    train_y = np.array(train_y)

    test_x = np.array(test_x, dtype=np.float32).reshape(
        len(test_x), 28, 28, 1) / 255.0
    test_y = np.array(test_y)

    idx = np.random.permutation(train_x.shape[0])
    train_x = train_x[idx]
    train_y = train_y[idx]

    val_idx = math.ceil(train_x.shape[0] * 0.1)
    val_x = train_x[:val_idx]
    val_y = train_y[:val_idx]

    train_x = train_x[val_idx:]
    train_y = train_y[val_idx:]

    return train_x, train_y, val_x, val_y, test_x, test_y


def input(img, batch_size):
    return np.tile(img, (batch_size, 1, 1, 1)).reshape(batch_size,
                                                       img_dim, img_dim, 1)


def dist(imgs, img, batch_size):
    if imgs.shape == (batch_size, img_dim, img_dim, 1):
        imgs = imgs.reshape(batch_size, 784)

    assert imgs.shape == (batch_size, 784)

    img = img.reshape((784,))

    imgs_pixels = imgs * 255.0
    img_pixels = img * 255.0

    diff = np.linalg.norm(imgs_pixels - img_pixels, axis=1)
    return np.mean(diff), np.std(diff)
