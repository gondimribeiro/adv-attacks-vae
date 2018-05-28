import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc

sns.set()


def metrics_auc(points, limits):
    (noise_dist, adv_dist) = points
    (ex_orig_target_dist, ex_orig_target_recon_dist,
     target_reconstruction_dist) = limits

    max_noise = max(noise_dist)
    min_dist = min(adv_dist)

    noise_dist += (max_noise,)
    adv_dist += (min_dist,)

    noise_dist += (ex_orig_target_dist,)
    adv_dist += (min_dist,)

    return auc(noise_dist, adv_dist)


def plot_metrics(plot_info, directory, file):
    (points, limits, Measure, bestC) = plot_info
    plt.figure()
    plt.axvline(x=limits[0], linewidth=2,
                color='cyan', label="Original - Target")
    plt.axhline(y=limits[1], linewidth=2,
                color='DarkOrange', label="Original rec. - Target")
    plt.axhline(y=limits[2], linewidth=2,
                color='red', label="Target rec. - Target")
    plt.scatter(points[0], points[1])

    if bestC is not None:
        plt.scatter(bestC['noise_dist'], bestC['adv_dist'],
                    color="red", label='Chosen Example')
    plt.ylabel("Adversarial rec. - Target")
    plt.xlabel("Distortion")

    plt.title('m = %f' % Measure)
    plt.legend()
    plt.savefig(directory + file.replace('.csv', '_metrics.png'))
    plt.close()


def calc_from_normalized(directory):
    data = np.load(directory + "/results/normalized_data.npy")

    metrics = []
    for i, d in enumerate(data):
        xs = d['diff_xs']
        zs = d['diff_zs']
        points = (xs, zs)
        limits = (1.0, 1.0, 0)

        m = metrics_auc(points, limits)
        plot_info = (points, limits, m, None)
        plot_metrics(plot_info, directory, '/results/exp_%d.csv' % i)
        metrics.append(m)

    return metrics
