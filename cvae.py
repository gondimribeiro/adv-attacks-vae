from model import Model

import tensorflow as tf


class ConvVAE(Model):
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
        tmp = tf.reshape(
            self.x_in,
            [self.batch_size] + self.config['input_shape']
        )

        for layer in self.config['conv_layers']:
            tmp = tf.layers.conv2d(
                tmp,
                layer['n_filters'],
                layer['filter_size'],
                layer['stride']
            )
            tmp = self.transfer_fct(tmp)

        tmp = tf.contrib.layers.flatten(tmp)
        for layer in self.config['dense_layers']:
            tmp = tf.layers.dense(tmp, layer)
            tmp = self.transfer_fct(tmp)

        z_mean = tf.layers.dense(tmp, self.z_size)
        z_log_var = tf.layers.dense(tmp, self.z_size)

        return z_mean, z_log_var

    def _generator_network(self):
        tmp = self.z
        for layer in self.config['gen_dense_layers']:
            tmp = tf.layers.dense(tmp, layer)
            tmp = self.transfer_fct(tmp)

        tmp = tf.reshape(tmp, [self.batch_size] +
                         self.config['gen_init_shape'])

        for layer in self.config['deconv_layers']:
            tmp = tf.layers.conv2d_transpose(
                tmp,
                layer['n_filters'],
                layer['filter_size'],
                layer['stride'],
                layer['pad']
            )
            tmp = self.transfer_fct(tmp)

        x_reconstr_mean = tf.layers.conv2d_transpose(
            tmp,
            self.config['input_shape'][-1],
            self.config['output']['filter_size'],
            1,
            padding='same',
        )
        x_reconstr_mean = tf.contrib.layers.flatten(x_reconstr_mean)

        if self.is_gaussian:
            x_reconstr_log_var = tf.layers.conv2d_transpose(
                tmp,
                self.config['input_shape'][-1],
                self.config['output']['filter_size'],
                1,
                padding='same'
            )
            x_reconstr_log_var = tf.contrib.layers.flatten(x_reconstr_log_var)
        else:
            x_reconstr_mean = tf.sigmoid(x_reconstr_mean)
            x_reconstr_log_var = None

        return x_reconstr_mean, x_reconstr_log_var
