from model import Model

import tensorflow as tf


class VAE(Model):
    def __init__(self, config):
        super().__init__(config)

    def _create_network(self):
        self.transfer_fct = self.config['transfer_fct']

        # Use recognition network to determine mean and log variance
        # of variables in latent space
        self.z_mean, self.z_log_var = self._recognition_network()

        # Draw one sample z of latent variables from Gaussian distribution
        eps = tf.random_normal(
            (self.batch_size, self.z_size),
            dtype=tf.float32
        )
        self.z = self.z_mean + tf.exp(0.5 * self.z_log_var) * eps

        # Use generator to determine mean and log variance of the reconstructed
        # sampled latent variable
        self.x_reconstr_mean, self.x_reconstr_log_var = \
            self._generator_network()

    def _recognition_network(self):
        tmp = self.x_in

        for layer in self.config['enc_layers']:
            tmp = tf.layers.dense(tmp, layer)
            tmp = self.transfer_fct(tmp)

        z_mean = tf.layers.dense(tmp, self.z_size)
        z_log_var = tf.layers.dense(tmp, self.z_size)

        return z_mean, z_log_var

    def _generator_network(self):
        tmp = self.z

        for layer in self.config['dec_layers']:
            tmp = tf.layers.dense(tmp, layer)
            tmp = self.transfer_fct(tmp)

        x_reconstr_mean = tf.layers.dense(tmp, self.img_size)

        if self.is_gaussian:
            x_reconstr_log_var = tf.layers.dense(tmp, self.img_size)
        else:
            x_reconstr_mean = tf.sigmoid(x_reconstr_mean)
            x_reconstr_log_var = None

        return x_reconstr_mean, x_reconstr_log_var
