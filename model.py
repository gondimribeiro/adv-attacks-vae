import tensorflow as tf
import numpy as np

import math
import time


c = - 0.5 * math.log(2 * math.pi)


def binary_crossentropy(t, o, eps=1e-8):
    return -(t * tf.log(o + eps) + (1.0 - t) * tf.log(1.0 - o + eps))


def log_normal2(x, mean, log_var, eps=1e-5):
    return c - log_var / 2 - tf.pow(x - mean, 2) / (2 * tf.exp(log_var) + eps)


def kl_divergence(mean1, log_var1, mean2, log_var2, eps=1e-8):
    mean_term = 0.5 * (tf.exp(log_var1) + tf.pow(mean1 - mean2, 2)) \
        / (tf.exp(log_var2) + eps)
    return mean_term + 0.5 * log_var2 - 0.5 * log_var1 - 0.5


class Model(object):

    def __init__(self, config):

        self.config = config
        self.learning_rate = config['learning_rate']
        self.adv_learning_rate = config['adv_learning_rate']
        self.batch_size = config['batch_size']
        self.A, self.B, self.n_chan = config['input_shape']
        self.img_size = self.A * self.B * self.n_chan
        self.z_size = config['z_size']
        self.is_gaussian = config['is_gaussian']
        self.is_attacking = config['is_attacking']

        self.mean = config['data_mean']
        self.std = config['data_std']
        self.max_pixel = config['data_max_pixel']

        # Input placeholders
        self.x = tf.placeholder(
            tf.float32,
            shape=(self.batch_size, self.A, self.B, self.n_chan),
            name='x'
        )

        # Adversarial noise
        self.noise = tf.Variable(
            tf.zeros([self.A, self.B, self.n_chan]),
            trainable=False,
            name='noise'
        )

        # Placeholders for adversarial attack
        self.noise_placeholder = tf.placeholder(
            tf.float32,
            shape=self.noise.get_shape(),
            name='noise_placeholder'
        )
        self.C = tf.placeholder(dtype=tf.float32, shape=[], name='C')

        self.adv_target_mean = tf.placeholder(
            tf.float32,
            [self.z_size],
            name=('adv_target_mean')
        )
        self.adv_target_log_var = tf.placeholder(
            tf.float32,
            [self.z_size],
            name=('adv_target_log_var')
        )
        self.adv_target_output = tf.placeholder(
            tf.float32,
            shape=(self.A, self.B, self.n_chan),
            name='adv_target_output'
        )

        # Operation to set noise and reset noise
        self.op_update_noise = self.noise.assign(self.noise_placeholder)
        self.op_reset_noise = self.noise.assign(
            tf.zeros([self.A, self.B, self.n_chan])
        )

        # Build graph
        self.x_in = self.x

        if self.is_attacking:
            self.x_in += self.noise
            self.x_in = tf.maximum(self.x_in, -self.mean / self.std)
            self.x_in = tf.minimum(
                self.x_in, (self.max_pixel - self.mean) / self.std)

        self.x_in = tf.reshape(self.x_in, [self.batch_size, -1])

        self._create_network()
        self._create_loss_optimizer()
        self._create_attack_optimizer()
        self._create_output_attack_optimizer()

        # Initializing the tensor flow variables
        init = tf.global_variables_initializer()

        # Launch the session
        self.sess = tf.InteractiveSession()
        self.sess.run(init)

        # Model saver
        self._saver = tf.train.Saver()

    def _create_network(self):
        pass

    def _compute_latent_loss(self):
        Lz = tf.reduce_sum(
            kl_divergence(
                self.z_mean,
                self.z_log_var,
                0.,
                1.),
            axis=1
        )

        return tf.reduce_mean(Lz)

    def _compute_reconstr_loss(self):
        flat_x_in = tf.reshape(self.x_in, [self.batch_size, -1])
        if self.is_gaussian:
            Lx = tf.reduce_sum(
                -log_normal2(
                    flat_x_in,
                    self.x_reconstr_mean,
                    self.x_reconstr_log_var,
                ),
                axis=1
            )
        else:
            Lx = tf.reduce_sum(
                binary_crossentropy(flat_x_in, self.x_reconstr_mean),
                1
            )

        return tf.reduce_mean(Lx)

    def _create_loss_optimizer(self):
        self.reconstr_Lx = self._compute_reconstr_loss()
        self.reconstr_Lz = self._compute_latent_loss()
        self.reconstr_loss = self.reconstr_Lx + self.reconstr_Lz

        def ClipIfNotNone(grad):
            if grad is None:
                return grad
            return tf.clip_by_value(grad, -1, 1)

        params = tf.trainable_variables()
        grads = tf.gradients(self.reconstr_loss, params)

        grads, _ = tf.clip_by_global_norm(grads, 3.)
        self.reconstr_grads = [ClipIfNotNone(grad) for grad in grads]

        self.recontr_optimizer = tf.train.AdamOptimizer(self.learning_rate)

        self.reconstr_update = self.recontr_optimizer.apply_gradients(
            zip(self.reconstr_grads, params)
        )

    def _create_attack_optimizer(self):
        self.adversarial_loss = self.C * tf.reduce_sum(self.noise * self.noise)

        self.adversarial_loss += (
            tf.reduce_sum(
                kl_divergence(
                    self.z_mean[0],
                    self.z_log_var[0],
                    self.adv_target_mean,
                    self.adv_target_log_var
                )
            )
        )

        adv_grads = tf.gradients(self.adversarial_loss, [self.noise])
        adv_grads, _ = tf.clip_by_global_norm(adv_grads, 3)
        self.adv_grad = tf.clip_by_value(adv_grads[0], -1, 1)

        self.adv_optimizer = tf.train.AdamOptimizer(self.adv_learning_rate)
        self.adv_update = self.adv_optimizer.apply_gradients(
            zip([self.adv_grad], [self.noise])
        )

    def _create_output_attack_optimizer(self):
        diff = self.adv_target_output - tf.reshape(
            self.x_reconstr_mean[0],
            shape=(self.A, self.B, self.n_chan)
        )

        self.adversarial_loss_output = (
            self.C *
            tf.reduce_sum(self.noise * self.noise)
        )

        self.adversarial_loss_output += tf.reduce_sum(diff * diff)

        adv_grads = tf.gradients(self.adversarial_loss_output, [self.noise])
        adv_grads, _ = tf.clip_by_global_norm(adv_grads, 3)
        self.adv_grad_output = tf.clip_by_value(adv_grads[0], -1, 1)

        self.adv_output_optimizer = tf.train.AdamOptimizer(
            self.adv_learning_rate
        )

        self.adv_output_update = self.adv_output_optimizer.apply_gradients(
            zip([self.adv_grad_output], [self.noise])
        )

    def batch_evaluate(self, X):
        assert X.shape[0] == self.batch_size

        loss = self.sess.run(
            self.reconstr_loss,
            feed_dict={self.x: X}
        )

        return loss

    def batch_transform(self, X):
        assert X.shape[0] == self.batch_size

        z_mean, z_log_var = self.sess.run(
            (self.z_mean, self.z_log_var),
            feed_dict={self.x: X}
        )

        return z_mean, z_log_var

    def batch_evaluate_attack(self, X, adv_target_mean=None,
                              adv_target_log_var=None, C=0,
                              adv_target_output=None):
        assert X.shape[0] == self.batch_size
        assert (
            (adv_target_mean is not None and adv_target_log_var is not None) or
            adv_target_output is not None
        )

        if adv_target_output is None:
            loss, grad = self.sess.run(
                (self.adversarial_loss, self.adv_grad),
                feed_dict={
                    self.x: X,
                    self.adv_target_mean: adv_target_mean,
                    self.adv_target_log_var: adv_target_log_var,
                    self.C: C
                }
            )
        else:
            loss, grad = self.sess.run(
                (self.adversarial_loss_output, self.adv_grad_output),
                feed_dict={
                    self.x: X,
                    self.adv_target_output: adv_target_output,
                    self.C: C
                }
            )

        return loss, grad

    def batch_optimize_attack(self, X,
                              adv_target_mean=None,
                              adv_target_log_var=None,
                              C=0,
                              adv_target_output=None,
                              num_iter=50):
        assert X.shape[0] == self.batch_size
        assert (
            (adv_target_mean is not None and adv_target_log_var is not None) or
            adv_target_output is not None
        )

        losses = []
        best_loss = float('Inf')
        init_noise = np.random.uniform(
            -1e-8,
            1e-8,
            size=self.config['input_shape']
        ).astype(np.float32)
        self.set_noise(init_noise)

        for i in range(num_iter):
            if adv_target_output is None:
                _, loss = self.sess.run(
                    (self.adv_update, self.adversarial_loss),
                    feed_dict={
                        self.x: X,
                        self.adv_target_mean: adv_target_mean,
                        self.adv_target_log_var: adv_target_log_var,
                        self.C: C
                    }
                )
            else:
                _, loss = self.sess.run(
                    (self.adv_output_update, self.adversarial_loss_output),
                    feed_dict={
                        self.x: X,
                        self.adv_target_output: adv_target_output,
                        self.C: C
                    }
                )

            losses.append(loss)
            if best_loss > loss:
                best_loss = loss
                best_noise = self.sess.run(self.noise)

        return best_noise, losses

    def batch_fit(self, X):
        assert X.shape[0] == self.batch_size

        _, loss = self.sess.run(
            (self.reconstr_update, self.reconstr_loss),
            feed_dict={self.x: X}
        )

        return loss

    def batch_reconstruct(self, X):
        assert X.shape[0] <= self.batch_size

        diff = None
        if X.shape[0] < self.batch_size:
            diff = self.batch_size - X.shape[0]
            X = np.vstack((X, np.zeros((diff,) + X.shape[1:])))

        X_reconstr = self.sess.run(
            self.x_reconstr_mean,
            feed_dict={self.x: X}
        )

        if diff is not None:
            X_reconstr = X_reconstr[:diff]

        return X_reconstr

    def reset_noise(self):
        self.sess.run(self.op_reset_noise)

    def set_noise(self, noise):
        assert noise.shape == tuple((self.A, self.B, self.n_chan))
        self.sess.run(
            self.op_update_noise,
            feed_dict={self.noise_placeholder: noise}
        )

    def evaluate(self, X):
        num_samples = X.shape[0]
        num_batches = math.floor(num_samples / self.batch_size)

        total_loss = 0
        for idx in range(0, num_samples, self.batch_size):
            batch_x = X[idx:(idx + self.batch_size)]
            if batch_x.shape[0] < self.batch_size:
                continue

            total_loss += self.batch_evaluate(batch_x)

        return total_loss / num_batches

    def fit(self, X, X_val, num_epochs, ckpt_path, random_sampling=False,
            img_path=None,  X_test=None, f_reconstruct=None, epoch_rec=10,
            verbose=True):
        num_samples = X.shape[0]
        num_batches = math.ceil(num_samples / self.batch_size)

        losses = []
        val_losses = []
        best_val = float('inf')
        self.reset_noise()
        for epoch in range(num_epochs):
            tic = time.time()
            total_loss = 0
            for i in range(num_batches):
                if random_sampling:
                    idx = np.random.choice(X.shape[0], self.batch_size)
                else:
                    idx = list(
                        range(
                            i * self.batch_size,
                            min((i + 1) * self.batch_size, X.shape[0])
                        )
                    )
                    if len(idx) < self.batch_size:
                        idx += np.random.choice(
                            X.shape[0],
                            self.batch_size - len(idx)
                        ).tolist()
                        continue
                total_loss += self.batch_fit(X[idx])

            val_losses.append(self.evaluate(X_val))
            if val_losses[-1] < best_val:
                best_val = val_losses[-1]
                self.save(ckpt_path + ("/model.epoch=%d.val_loss=%.4f.ckpt"
                                       % (epoch + 1, best_val)))

            if (epoch % epoch_rec == 0 and X_test is not None and
                    img_path is not None and f_reconstruct is not None):
                f_reconstruct(
                    self,
                    X_test,
                    img_path + ('/rec.epoch=%d.png' % (epoch + 1)),
                    normalize=self.is_gaussian
                )

            losses.append(total_loss / num_batches)
            if verbose:
                print('Epoch %d/%d: loss=%f, elapsed=%.4fs' %
                      (epoch + 1, num_epochs, losses[-1], time.time() - tic))
                print('\tval_loss=%f' % val_losses[-1])

        return losses, val_losses

    def load(self, ckpt_path):
        self._saver.restore(self.sess, ckpt_path)

    def save(self, ckpt_path):
        self._saver.save(self.sess, ckpt_path)

    def close(self):
        self.sess.close()
