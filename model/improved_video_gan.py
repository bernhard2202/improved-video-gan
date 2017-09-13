from __future__ import division, print_function

import time
import numpy as np
import tensorflow as tf

from utils.layers import conv3d_transpose, dis_block, linear
from utils.utils import sampleBatch, saveGIFBatch


class ImprovedVideoGAN(object):
    def __init__(self,
                 input_batch,
                 batch_size=64,
                 frame_size=32,
                 crop_size=64,
                 learning_rate=0.0002,
                 z_dim=100,
                 beta1=0.5,
                 alpha1=0.1,
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.z_dim = z_dim
        self.frame_size = frame_size
        self.videos = input_batch
        self.alpha1 = alpha1
        self.build_model()

    def generator(self, z):
        with tf.variable_scope('g_') as vs:
            """ LINEAR BLOCK """
            self.z_, _, _ = linear(z, 512 * 4 * 4 * 2, 'g_f_h0_lin', with_w=True)
            self.fg_h0 = tf.reshape(self.z_, [-1, 2, 4, 4, 512])
            self.fg_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h0, scope='g_f_bn0'), name='g_f_relu0')
            add_activation_summary(self.fg_h0)

            """ CONV BLOCK 1 """
            self.fg_h1 = conv3d_transpose(self.fg_h0, 512, [self.batch_size, 4, 8, 8, 256], name='g_f_h1')
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)

            """ CONV BLOCK 2 """
            self.fg_h2 = conv3d_transpose(self.fg_h1, 256, [self.batch_size, 8, 16, 16, 128], name='g_f_h2')
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)

            """ CONV BLOCK 3 """
            self.fg_h3 = conv3d_transpose(self.fg_h2, 128, [self.batch_size, 16, 32, 32, 64], name='g_f_h3')
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)

            """ CONV BLOCK 5 """
            self.fg_h4 = conv3d_transpose(self.fg_h3, 64, [self.batch_size, 32, 64, 64, 3], name='g_f_h4')
            self.fg_fg = tf.nn.tanh(self.fg_h4, name='g_f_actvcation')

        variables = tf.contrib.framework.get_variables(vs)
        return self.fg_fg, variables

    def discriminator(self, video, reuse=False):
        with tf.variable_scope('d_', reuse=reuse) as vs:
            initial_dim = 64
            """ CONV BLOCK 1 """
            d_h0 = dis_block(video, 3, initial_dim, 'block1', reuse=reuse)
            """ CONV BLOCK 2 """
            d_h1 = dis_block(d_h0, initial_dim, initial_dim * 2, 'block2', reuse=reuse)
            """ CONV BLOCK 3 """
            d_h2 = dis_block(d_h1, initial_dim * 2, initial_dim * 4, 'block3', reuse=reuse)
            """ CONV BLOCK 4 """
            d_h3 = dis_block(d_h2, initial_dim * 4, initial_dim * 8, 'block4', reuse=reuse)
            """ CONV BLOCK 5 """
            d_h4 = dis_block(d_h3, initial_dim * 8, 1, 'block5', reuse=reuse, normalize=False)
            """ LINEAR BLOCK """
            d_h5 = linear(tf.reshape(d_h4, [self.batch_size, -1]), 1)
        variables = tf.contrib.framework.get_variables(vs)
        return d_h5, variables

    def build_model(self):
        print("Setting up model...")
        self.z_vec = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name="z")

        tf.summary.histogram("z", self.z_vec)
        self.videos_fake, self.generator_variables = self.generator(self.z_vec)

        self.d_real, self.discriminator_variables = self.discriminator(self.videos, reuse=False)
        self.d_fake, _ = self.discriminator(self.videos_fake, reuse=True)

        self.g_cost = -tf.reduce_mean(self.d_fake)
        self.d_cost = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)

        tf.summary.scalar("g_cost", self.g_cost)
        tf.summary.scalar("d_cost", self.d_cost)

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )

        dim = self.frame_size * self.crop_size * self.crop_size * 3

        vid = tf.reshape(self.videos, [self.batch_size, dim])
        fake = tf.reshape(self.videos_fake, [self.batch_size, dim])
        differences = fake - vid
        interpolates = vid + (alpha * differences)
        d_hat, _ = self.discriminator(
            tf.reshape(interpolates, [self.batch_size, self.frame_size, self.crop_size, self.crop_size, 3]), reuse=True)
        gradients = tf.gradients(d_hat, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.d_cost_final = self.d_cost + 10 * gradient_penalty

        tf.summary.scalar("d_cost_penalized", self.d_cost_final)

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.d_cost_final, var_list=self.discriminator_variables)
            self.g_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.g_cost, var_list=self.generator_variables)

        print("\nTrainable variables for generator:")
        for var in self.generator_variables:
            print(var.name)
        print("\nTrainable variables for discriminator:")
        for var in self.discriminator_variables:
            print(var.name)

        self.sample = sampleBatch(self.videos_fake, self.batch_size)
        self.summary_op = tf.summary.merge_all()

    def get_feed_dict(self):
        batch_z = np.random.normal(0, 1.0, size=[self.batch_size, self.z_dim]).astype(np.float32)
        feed_dict = {self.z_vec: batch_z}
        return feed_dict

    def train(self,
              session,
              step,
              summary_writer=None,
              log_summary=False,
              sample_dir=None,
              generate_sample=False):
        if log_summary:
            start_time = time.time()

        critic_itrs = self.critic_iterations

        for critic_itr in range(critic_itrs):
            session.run(self.d_adam, feed_dict=self.get_feed_dict())

        feed_dict = self.get_feed_dict()
        session.run(self.g_adam, feed_dict=feed_dict)

        if log_summary:
            g_loss_val, d_loss_val, summary = session.run([self.g_cost, self.d_cost_final, self.summary_op],
                                                          feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            print("Time: %g/itr, Step: %d, generator loss: %g, discriminator_loss: %g" % (
                time.time() - start_time, step, g_loss_val, d_loss_val))

        if generate_sample:
            vid_sample = session.run(self.sample, feed_dict=feed_dict)
            saveGIFBatch(vid_sample, sample_dir, 'vid_%d' % step)


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + '/gradient', grad)
