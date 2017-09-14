from __future__ import division, print_function

import time
import tensorflow as tf

from utils.layers import conv2d, conv3d_transpose, dis_block, linear
from utils.utils import sampleBatch, saveGIFBatch, write_image


class ImprovedVideoGANFuture(object):
    def __init__(self,
                 input_batch,
                 batch_size=64,
                 frame_size=32,
                 crop_size=64,
                 learning_rate=0.0002,
                 beta1=0.5,
                 critic_iterations=5):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.frame_size = frame_size
        self.videos = input_batch
        self.build_model()

    def generator(self, img_batch):
        with tf.variable_scope('g_') as vs:
            """ -----------------------------------------------------------------------------------
            ENCODER 
            ----------------------------------------------------------------------------------- """
            self.en_h0 = conv2d(img_batch, 3, 128, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv1")
            self.en_h0 = tf.nn.relu(tf.contrib.layers.batch_norm(self.en_h0))
            add_activation_summary(self.en_h0)
            print(self.en_h0.get_shape().as_list())

            self.en_h1 = conv2d(self.en_h0, 128, 256, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv2")
            self.en_h1 = tf.contrib.layers.batch_norm(self.en_h1, scope="enc_bn2")
            self.en_h1 = tf.nn.relu(self.en_h1)
            add_activation_summary(self.en_h1)
            print(self.en_h1.get_shape().as_list())

            self.en_h2 = conv2d(self.en_h1, 256, 512, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv3")
            self.en_h2 = tf.contrib.layers.batch_norm(self.en_h2, scope="enc_bn3")
            self.en_h2 = tf.nn.relu(self.en_h2)
            add_activation_summary(self.en_h2)
            print(self.en_h2.get_shape().as_list())

            self.en_h3 = conv2d(self.en_h2, 512, 1024, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv4")
            self.en_h3 = tf.contrib.layers.batch_norm(self.en_h3, scope="enc_bn4")
            self.en_h3 = tf.nn.relu(self.en_h3)
            add_activation_summary(self.en_h3)
            print(self.en_h3.get_shape().as_list())

            """ -----------------------------------------------------------------------------------
            GENERATOR 
            ----------------------------------------------------------------------------------- """
            self.z_ = tf.reshape(self.en_h3, [self.batch_size, 2, 4, 4, 512])
            print(self.z_.get_shape().as_list())

            self.fg_h1 = conv3d_transpose(self.z_, 512, [self.batch_size, 4, 8, 8, 256], name='g_f_h1')
            self.fg_h1 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h1, scope='g_f_bn1'), name='g_f_relu1')
            add_activation_summary(self.fg_h1)
            print(self.fg_h1.get_shape().as_list())

            self.fg_h2 = conv3d_transpose(self.fg_h1, 256, [self.batch_size, 8, 16, 16, 128], name='g_f_h2')
            self.fg_h2 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h2, scope='g_f_bn2'), name='g_f_relu2')
            add_activation_summary(self.fg_h2)
            print(self.fg_h2.get_shape().as_list())

            self.fg_h3 = conv3d_transpose(self.fg_h2, 128, [self.batch_size, 16, 32, 32, 64], name='g_f_h3')
            self.fg_h3 = tf.nn.relu(tf.contrib.layers.batch_norm(self.fg_h3, scope='g_f_bn3'), name='g_f_relu3')
            add_activation_summary(self.fg_h3)
            print(self.fg_h3.get_shape().as_list())

            self.fg_h4 = conv3d_transpose(self.fg_h3, 64, [self.batch_size, 32, 64, 64, 3], name='g_f_h4')
            self.fg_fg = tf.nn.tanh(self.fg_h4, name='g_f_actvcation')
            print(self.fg_fg.get_shape().as_list())

            gen_reg = tf.reduce_mean(tf.square(img_batch - self.fg_fg[:, 0, :, :, :]))

        variables = tf.contrib.framework.get_variables(vs)
        return self.fg_fg, gen_reg, variables

    def discriminator(self, video, reuse=False):
        with tf.variable_scope('d_', reuse=reuse) as vs:
            initial_dim = 64
            d_h0 = dis_block(video, 3, initial_dim, 'block1', reuse=reuse)
            d_h1 = dis_block(d_h0, initial_dim, initial_dim * 2, 'block2', reuse=reuse)
            d_h2 = dis_block(d_h1, initial_dim * 2, initial_dim * 4, 'block3', reuse=reuse)
            d_h3 = dis_block(d_h2, initial_dim * 4, initial_dim * 8, 'block4', reuse=reuse)
            d_h4 = dis_block(d_h3, initial_dim * 8, 1, 'block5', reuse=reuse, normalize=False)
            d_h5 = linear(tf.reshape(d_h4, [self.batch_size, -1]), 1)
        variables = tf.contrib.framework.get_variables(vs)
        return d_h5, variables

    def build_model(self):
        print("Setting up model...")

        self.input_images = tf.placeholder(tf.float32, [self.batch_size, self.crop_size, self.crop_size, 3])
        self.videos_fake, self.gen_reg, self.generator_variables = self.generator(self.input_images)

        self.d_real, self.discriminator_variables = self.discriminator(self.videos, reuse=False)
        self.d_fake, _ = self.discriminator(self.videos_fake, reuse=True)

        self.g_cost_pure = -tf.reduce_mean(self.d_fake)

        #self.g_cost = self.g_cost_pure + 1000 * self.gen_reg

        self.d_cost = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)

        tf.summary.scalar("g_cost_pure", self.g_cost_pure)
        tf.summary.scalar("g_cost_regularizer", self.gen_reg)
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
        d_hat, _ = self.discriminator(tf.reshape(interpolates, [self.batch_size, self.frame_size, self.crop_size,
                                                                self.crop_size, 3]), reuse=True)
        gradients = tf.gradients(d_hat, [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.d_cost_final = self.d_cost + 10 * gradient_penalty

        tf.summary.scalar("d_cost_penalized", self.d_cost_final)

        self.d_adam, self.g_adam = None, None
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.d_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.d_cost_final, var_list=self.discriminator_variables)
            self.g_adam_gan = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.g_cost_pure, var_list=self.generator_variables)
            self.g_adam_first = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.gen_reg, var_list=self.generator_variables)

        self.sample = sampleBatch(self.videos_fake, self.batch_size)
        self.summary_op = tf.summary.merge_all()

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        for grad, var in grads:
            add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    def get_feed_dict(self, session):
        images = session.run(self.videos)[:, 0, :, :, :]
        feed_dict = {self.input_images: images}
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
            session.run(self.d_adam, feed_dict=self.get_feed_dict(session))

        feed_dict = self.get_feed_dict(session)
        session.run(self.g_adam_gan, feed_dict=feed_dict)
        session.run(self.g_adam_first, feed_dict=feed_dict)

        if log_summary:
            g_loss_pure, g_reg, d_loss_val, summary = session.run(
                [self.g_cost_pure, self.gen_reg, self.d_cost, self.summary_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            print("Time: %g/itr, Step: %d, generator loss: (%g + %g), discriminator_loss: %g" % (
                time.time() - start_time, step, g_loss_pure, g_reg, d_loss_val))

        if generate_sample:
            images = session.run(self.videos)[:, 0, :, :, :]
            write_image(images, sample_dir, 'vid_%d_f0.jpg' % step, rows=5)
            vid_sample = session.run(self.sample, feed_dict={self.input_images: images})
            saveGIFBatch(vid_sample, sample_dir, 'vid_%d_future' % step)


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + '/gradient', grad)
