import matplotlib
matplotlib.use("Agg")

import argparse
import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import scipy
import sys
import tensorflow as tf
import time

from draw import DRAW
from cvae import ConvVAE
from vae import VAE

import celeba_utils
import svhn_utils
import mnist_utils


def print_log(s='', verbose=True):
    LOG_FP.write(s + '\n')
    if verbose:
        print(s)


def optimize_noise(vae, orig_img, target_img, C, attack_output=False,
                   bfgs=True):
    vae_x = orig_img.reshape(-1, data_utils.img_dim,
                             data_utils.img_dim, data_utils.n_chan)

    if not attack_output:
        adv_mean, adv_log_var = vae.batch_transform(
            data_utils.input(target_img, batch_size))

        adv_mean = adv_mean[0]
        adv_log_var = adv_log_var[0]
        adv_target_output = None
    else:
        adv_mean = None
        adv_log_var = None
        adv_target_output = target_img

    if bfgs:
        x, adv_loss = optimize_noise_bfgs(vae, orig_img, vae_x, adv_mean,
                                          adv_log_var, adv_target_output, C)
    else:
        x, adv_loss = vae.batch_optimize_attack(
            data_utils.input(vae_x, batch_size),
            adv_mean,
            adv_log_var,
            C,
            adv_target_output,
            num_iter=NUM_ITER
        )

    return x, adv_loss


def optimize_noise_bfgs(vae, orig_img, vae_x, adv_mean, adv_log_var,
                        adv_target_output, C):
    init_noise = np.random.uniform(
        -1e-8,
        1e-8,
        size=(data_utils.img_dim, data_utils.img_dim, data_utils.n_chan)
    ).astype(np.float32)

    adv_loss = []

    def fmin_func(noise):
        vae.set_noise(
            noise.reshape(
                data_utils.img_dim, data_utils.img_dim, data_utils.n_chan
            )
        )
        loss, grad = vae.batch_evaluate_attack(
            data_utils.input(vae_x, batch_size),
            adv_mean,
            adv_log_var,
            C,
            adv_target_output
        )

        adv_loss.append(loss)
        return float(loss), grad.flatten().astype(np.float64)

    # Noise bounds (pixels cannot exceed 0-1)
    # ToDo: correct limits
    bounds = zip(
        -data_utils.data_mean / data_utils.data_std - orig_img.flatten(),
        (data_utils.max_pixel - data_utils.data_mean) / data_utils.data_std -
        orig_img.flatten()
    )
    bounds = [sorted(x) for x in bounds]

    # L-BFGS-B optimization to find adversarial noise
    x, f, d = scipy.optimize.fmin_l_bfgs_b(
        fmin_func,
        x0=init_noise,
        bounds=bounds,
        m=25,
        factr=10
    )

    x = x.reshape(data_utils.img_dim, data_utils.img_dim, data_utils.n_chan)

    return x, adv_loss


def adv_test(vae, orig_img, target_img, C, plot=True):
    vae.reset_noise()

    # Original and target reconstruction
    original_reconstructions = vae.batch_reconstruct(
        data_utils.input(orig_img, batch_size))
    target_reconstructions = vae.batch_reconstruct(
        data_utils.input(target_img, batch_size))

    orig_recon_dist, orig_recon_dist_std = data_utils.dist(
        original_reconstructions, orig_img, batch_size)
    target_recon_dist, target_recon_dist_std = data_utils.dist(
        target_reconstructions, target_img, batch_size)
    orig_target_recon_dist, orig_target_recon_dist_std = data_utils.dist(
        original_reconstructions, target_img, batch_size)
    target_orig_recon_dist, target_orig_recon_dist_std = data_utils.dist(
        target_reconstructions, orig_img, batch_size)

    x, adv_loss = \
        optimize_noise(vae, orig_img, target_img, C, ATTACK_OUTPUT, BFGS)

    adv_input = x + orig_img

    # Adversarial reconstruction
    vae.reset_noise()
    adv_imgs = vae.batch_reconstruct(data_utils.input(adv_input, batch_size))
    orig_dist, orig_dist_std = data_utils.dist(
        adv_imgs, orig_img, batch_size)
    adv_dist, adv_dist_std = data_utils.dist(
        adv_imgs, target_img, batch_size)
    recon_dist, recon_dist_std = data_utils.dist(
        adv_imgs, adv_input, batch_size)

    # Plotting results
    if plot:
        data_utils.show(orig_img, True, 1, "Original")
        data_utils.show(
            original_reconstructions[0], True, 2, "Original rec.")
        data_utils.show(adv_input, True, 3, "Adversarial")
        data_utils.show(target_img, True, 4, "Target")
        data_utils.show(adv_imgs[0], True, 5, "Adversarial rec.")
        data_utils.show(x, True, 6, "Distortion")
        plt.savefig(model_dir + ('/results/exp_%d_imgs.png' % FIG_COUNT))
        plt.close()

        plt.figure()
        plt.scatter(range(len(adv_loss)), adv_loss)
        plt.ylabel('adv_loss')
        plt.xlabel('iter')
        plt.savefig(model_dir + ('/results/exp_%d_loss.png' % FIG_COUNT))
        plt.close()

        np.save(model_dir + ('/results/exp_%d_noise' % FIG_COUNT), x)

    orig_target_dist = np.linalg.norm(orig_img - target_img)

    returns = (
        np.linalg.norm(x),
        orig_dist,
        orig_dist_std,
        adv_dist,
        adv_dist_std,
        orig_recon_dist,
        orig_recon_dist_std,
        target_recon_dist,
        target_recon_dist_std,
        recon_dist,
        recon_dist_std,
        orig_target_dist,
        orig_target_recon_dist,
        orig_target_recon_dist_std,
        target_orig_recon_dist,
        target_orig_recon_dist_std,
        adv_loss[-1]
    )

    return returns


def orig_adv_dist(vae, orig_img=None, target_img=None, plot=False, bestC=None):
    if orig_img is None:
        orig_img = np.random.randint(0, len(test_x))
    if target_img is None:
        target_img = orig_img
        while np.array_equal(target_img, orig_img):
            target_img = np.random.randint(0, len(test_x))

    noise_dist = []
    orig_dist = []
    orig_dist_std = []
    adv_dist = []
    adv_dist_std = []
    target_recon_dist = []
    target_recon_dist_std = []
    recon_dist = []
    recon_dist_std = []
    orig_target_dist = []
    orig_target_recon_dist = []
    orig_target_recon_dist_std = []
    target_orig_recon_dist = []
    target_orig_recon_dist_std = []
    adv_loss = []

    C = np.logspace(-20, 20, NUM_POINTS, base=2, dtype=np.float32)
    C = np.concatenate(([0], C))

    for c in C:
        noise, od, ods, ad, ads, ore, ores, tre, tres, recd, recs, otd, otrd, \
            otrds, tord, tords, advl = adv_test(
                vae, test_x[orig_img], test_x[target_img], C=c, plot=False)
        noise_dist.append(noise)
        orig_dist.append(od)
        orig_dist_std.append(ods)
        adv_dist.append(ad)
        adv_dist_std.append(ads)
        target_recon_dist.append(tre)
        target_recon_dist_std.append(tres)
        recon_dist.append(recd)
        recon_dist_std.append(recs)
        orig_target_dist.append(otd)
        orig_target_recon_dist.append(otrd)
        orig_target_recon_dist_std.append(otrds)
        target_orig_recon_dist.append(tord)
        target_orig_recon_dist_std.append(tords)
        adv_loss.append(advl)

    noise_dist = np.array(noise_dist)
    orig_dist = np.array(orig_dist)
    orig_dist_std = np.array(orig_dist_std)
    adv_dist = np.array(adv_dist)
    adv_dist_std = np.array(adv_dist_std)
    target_recon_dist = np.array(target_recon_dist)
    target_recon_dist_std = np.array(target_recon_dist_std)
    recon_dist = np.array(recon_dist)
    recon_dist_std = np.array(recon_dist_std)
    orig_target_dist = np.array(orig_target_dist)
    orig_target_recon_dist = np.array(orig_target_recon_dist)
    orig_target_recon_dist_std = np.array(orig_target_recon_dist_std)
    target_orig_recon_dist = np.array(target_orig_recon_dist)
    target_orig_recon_dist_std = np.array(target_orig_recon_dist_std)
    adv_loss = np.array(adv_loss)

    if bestC is None:
        tmp_idx = adv_dist <= np.mean(adv_dist)
        bestC = np.atleast_1d(
            np.atleast_1d(C[tmp_idx])[np.argmax(adv_dist[tmp_idx])]
        )[0]

    ex_noise, _, _, ex_adv_dist, _, orig_reconstruction_dist, _, \
        target_reconstruction_dist, _, _, _, ex_orig_target_dist, \
        ex_orig_target_recon_dist, _, _, _, _ = adv_test(
            vae, test_x[orig_img], test_x[target_img], C=bestC, plot=plot)
    print_log(
        "orig_img=%d, target_img=%d, bestC=%f, adv_dist=%f, noise_norm=%f"
        % (orig_img, target_img, bestC, ex_adv_dist, np.linalg.norm(ex_noise))
    )

    if plot:
        plt.figure()
        plt.axvline(x=ex_orig_target_dist, linewidth=2,
                    color='cyan', label="Original - Target")
        plt.axhline(y=ex_orig_target_recon_dist, linewidth=2,
                    color='DarkOrange', label="Original rec. - Target")
        plt.axhline(y=target_reconstruction_dist, linewidth=2,
                    color='red', label="Target rec. - Target")
        plt.scatter(noise_dist, adv_dist)
        plt.scatter([ex_noise], [ex_adv_dist], color="red")
        plt.ylabel("Adversarial rec. - Target")
        plt.xlabel("Distortion")
        plt.legend()
        plt.savefig(model_dir + ('/results/exp_%d.png' % FIG_COUNT))
        plt.close()

        # Adversarial Loss
        plt.figure()
        plt.scatter(noise_dist, adv_loss)
        plt.ylabel("Adversarial Loss")
        plt.xlabel("Distortion")
        plt.savefig(model_dir + ('/results/exp_%d_adv_loss.png' % FIG_COUNT))
        plt.close()

    df = pd.DataFrame(
        {
            'orig_img': orig_img,
            'target_img': target_img,
            'bestC': bestC,
            'orig_reconstruction_dist': orig_reconstruction_dist,
            'target_reconstruction_dist': target_reconstruction_dist,
            'noise_dist': noise_dist,
            'orig_dist': orig_dist,
            'orig_dist_std': orig_dist_std,
            'adv_dist': adv_dist,
            'adv_dist_std': adv_dist_std,
            'target_recon_dist': target_recon_dist,
            'target_recon_dist_std': target_recon_dist_std,
            'recon_dist': recon_dist,
            'recon_dist_std': recon_dist_std,
            'orig_target_dist': orig_target_dist,
            'orig_target_recon_dist': orig_target_recon_dist,
            'orig_target_recon_dist_std': orig_target_recon_dist_std,
            'target_orig_recon_dist': target_orig_recon_dist,
            'target_orig_recon_dist_std': target_orig_recon_dist_std,
            'C': C
        }
    )

    with open(model_dir + ("/results/exp_%d.csv" % FIG_COUNT), 'w') as fp:
        df.to_csv(fp)


def run_experiments(pairs):
    global FIG_COUNT

    n = len(pairs)
    for i in range(STARTING_FIG, n):
        FIG_COUNT = i
        start_time = time.time()
        print_log("----------------------------------------")
        print_log("Experiment %d/%d" % (i + 1, n))
        orig_adv_dist(
            vae,
            orig_img=pairs[i][0],
            target_img=pairs[i][1],
            plot=True
        )
        print_log("\tTime %f sec" % (time.time() - start_time))
        print_log()
        gc.collect()


if __name__ == "__main__":
    np.random.seed(0)
    tf.set_random_seed(0)

    sns.set()

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./celeba/draw')
    parser.add_argument('--num_attacks', type=int, default=20)
    parser.add_argument('--num_iter', type=int, default=3000)
    parser.add_argument('--num_points', type=int, default=50)
    parser.add_argument('--starting_fig', type=int, default=0)
    parser.add_argument('--attack_output', dest='attack_output',
                        action='store_true')
    parser.set_defaults(attack_output=False)

    parser.add_argument('--bfgs', dest='bfgs', action='store_true')
    parser.set_defaults(bfgs=False)

    # Parser
    args, unknown = parser.parse_known_args()

    if len(unknown) > 0:
        print("Invalid arguments %s" % unknown)
        parser.print_log_help()
        sys.exit()

    args = vars(parser.parse_args())
    model_dir = args['dir']

    # Networks architecture
    with open(model_dir + '/architecture.pkl', 'rb') as fp:
        config = pickle.load(fp)

    LOG_FP = open(model_dir + '/attacker.log', 'a+')

    if config['dataset'] == 'celeba':
        data_utils = celeba_utils
        train_x, val_x, test_x = data_utils.load_data(config['is_gaussian'])
        print_log("CelebA loaded")
    elif config['dataset'] == 'svhn':
        data_utils = svhn_utils
        train_x, _, val_x, _, test_x, _ = data_utils.load_data(
            config['is_gaussian'])
        print_log("SVHN loaded")
    elif config['dataset'] == 'mnist':
        data_utils = mnist_utils
        train_x, _, val_x, _, test_x, _ = data_utils.load_data(
            config['is_gaussian'])
        print_log("MNIST loaded")
    else:
        sys.exit("No dataset %s" % args['dataset'])

    print_log("Training samples %d" % train_x.shape[0])
    print_log("Validation samples %d" % val_x.shape[0])
    print_log("Test samples %d" % test_x.shape[0])
    print_log("----------------------------------------")

    pairs = []
    for i in range(args['num_attacks']):
        orig_img = np.random.randint(0, len(test_x))
        target_img = orig_img
        while np.array_equal(target_img, orig_img):
            target_img = np.random.randint(0, len(test_x))
        pairs.append([orig_img, target_img])

    # Load model
    tf.reset_default_graph()

    config['is_attacking'] = True
    if config['architecture'] == 'draw':
        vae = DRAW(config)
        print_log("----------------------------------------")
        print_log("DRAW graph loaded")
    elif config['architecture'] == 'cvae':
        vae = ConvVAE(config)
        print_log("----------------------------------------")
        print_log("ConvVAE graph loaded")
    elif config['architecture'] == 'vae':
        vae = VAE(config)
        print_log("----------------------------------------")
        print_log("VAE graph loaded")
    else:
        sys.exit("No architecture %s" % config['architecture'])

    tf.get_default_graph().finalize()
    vae.load(tf.train.latest_checkpoint(model_dir + '/model/'))

    batch_size = vae.batch_size
    ATTACK_OUTPUT = args['attack_output']
    BFGS = args['bfgs']
    NUM_ITER = args['num_iter']
    FIG_COUNT = None
    STARTING_FIG = args['starting_fig']
    NUM_POINTS = args['num_points']
    run_experiments(pairs)
    vae.close()
