import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

sns.set()


def normalize_data(dir, n, title=''):
    orig_dist = []
    orig_dist_std = []
    adv_dist = []
    adv_dist_std = []
    noise_dist = []
    recon_dist = []
    orig_target_dist = []
    target_recon_dist = []
    target_recon_dist_std = []
    orig_target_recon_dist = []
    orig_target_recon_dist_std = []
    C = []

    for i in range(n):
        df = pd.read_csv(dir + "/results/exp_" + str(i) + ".csv")
        orig_dist.append(df['orig_dist'].values)
        orig_dist_std.append(df['orig_dist_std'].values)
        adv_dist.append(df['adv_dist'].values)
        adv_dist_std.append(df['adv_dist_std'].values)
        noise_dist.append(df['noise_dist'].values)
        recon_dist.append(df['recon_dist'].values)
        target_recon_dist.append(df['target_recon_dist'].values)
        target_recon_dist_std.append(df['target_recon_dist'].values)
        orig_target_dist.append(df['orig_target_dist'].values)
        orig_target_recon_dist.append(df['orig_target_recon_dist'].values)
        orig_target_recon_dist_std.append(
            df['orig_target_recon_dist_std'].values)
        C.append(df['C'].values)

    normalized_data = []

    for i in range(n):
        xs = noise_dist[i] / orig_target_dist[i]
        zs = (
            (adv_dist[i] - target_recon_dist[i]) /
            (orig_target_recon_dist[i] - target_recon_dist[i])
        )
        zsds = (
            np.sqrt(adv_dist_std[i]**2.0 + target_recon_dist_std[i]**2.0) /
            np.sqrt(orig_target_recon_dist_std[i]
                    ** 2.0 + target_recon_dist_std[i]**2.)
        )

        idx = np.argsort(xs)
        xs = xs[idx]
        zs = zs[idx]
        zsds = zsds[idx]

        (xs, zs, zsds) = zip(*[(x, z, zd) for (x, z, zd)
                               in zip(xs, zs, zsds) if x >= 0 and x <= 1])

        normalized_data.append(dict())
        normalized_data[i]['diff_xs'] = xs
        normalized_data[i]['diff_zs'] = zs
        normalized_data[i]['diff_zsds'] = zsds

        ys = noise_dist[i] / orig_target_dist[i]
        zs = recon_dist[i] / orig_target_recon_dist[i]

        idx = np.argsort(ys)
        ys = ys[idx]
        zs = zs[idx]

        (ys, zs) = zip(*[(y, z)
                         for (y, z) in zip(ys, zs) if y >= 0 and y <= 1])

        normalized_data[i]['close_ys'] = ys
        normalized_data[i]['close_zs'] = zs

    np.save(dir + "/results/normalized_data", normalized_data)
    plot_data(normalized_data, title, dir)


def plot_data(data, title='', dir=None, error_bars=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, d in enumerate(data):
        xs = d['diff_xs']
        zs = d['diff_zs']
        zsds = d['diff_zsds']

        npoints = len(xs)
        ax.plot(xs, [i] * npoints, zs, alpha=0.5)

        if error_bars:
            for j in range(len(xs)):
                x = np.array([xs[j], xs[j]])
                y = np.array([i, i])
                z = np.array([zs[j], zs[j]])
                zerror = np.array([-zsds[j], zsds[j]])
                ax.plot(x, y, z + zerror, marker="_", alpha=0.25)

    ax.set_xlabel('Distortion')
    ax.set_xlim3d(0, 1)
    ax.set_ylabel('Experiment')
    ax.set_zlabel('Adversarial rec. - Target')
    ax.set_zlim3d(0, 1)
    fig.set_figwidth(8)
    fig.set_figheight(6)
    plt.title(title)
    if dir is None:
        plt.show()
    else:
        plt.savefig(dir + '/results/diff_plot.png')
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, d in enumerate(data):
        ys = d['close_ys']
        zs = d['close_zs']

        npoints = len(ys)
        ax.plot([i] * npoints, ys, zs, alpha=0.5)

    ax.set_ylabel('Distortion')
    ax.set_ylim3d(0, 1)
    ax.set_xlabel('Experiment')
    ax.set_zlabel('Adversarial rec. - Adversarial input')
    ax.set_zlim3d(0, 1)
    fig.set_figwidth(9)
    fig.set_figheight(6)
    plt.title(title)

    if dir is None:
        plt.show()
    else:
        plt.savefig(dir + '/results/close_plot.png')

    plt.close()
