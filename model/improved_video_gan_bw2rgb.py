from __future__ import division, print_function

import time
import tensorflow as tf

from utils.layers import conv3d_transpose, conv2d, dis_block, linear
from utils.utils import saveGIFBatch, sampleBatch


def get_frame_image_(video, frame, batch_size, col=5, row=5):
    video = tf.expand_dims(video[:, :, :, frame], 3)
    video = tf.image.convert_image_dtype(tf.div(tf.add(video, 1.0), 2.0), tf.uint8)
    video = [image for image in tf.split(video, batch_size, axis=0)]
    rows = []
    for i in range(row):
        rows.append(tf.concat(video[col * i + 0:col * i + col], 2))
    image = tf.concat(rows, 1)
    return tf.image.encode_jpeg(tf.squeeze(image, [0]), format="grayscale", quality=100)


def sampleBatch_(samples, batch_size, col=5, row=5, frames=32):
    frames = [get_frame_image_(samples, i, batch_size, col, row) for i in range(frames)]
    return frames


def rgb_to_grey(video, frames=32):
    grey_frames = [tf.expand_dims(video[:, i, :, :, 0], 3) * 0.21 + tf.expand_dims(video[:, i, :, :, 1],
                                                                                   3) * 0.72 + tf.expand_dims(
        video[:, i, :, :, 2], 3) * 0.07 for i in range(frames)]
    return tf.concat(grey_frames, 3)


class ImprovedVideoGANCol(object):
    def __init__(self,
                 input_batch,
                 batch_size=64,
                 frame_size=32,
                 crop_size=64,
                 learning_rate=0.0002,
                 beta1=0.5,
                 critic_iterations=4):
        self.critic_iterations = critic_iterations
        self.crop_size = crop_size
        self.beta1 = beta1
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.frame_size = frame_size
        self.videos = input_batch
        self.build_model()

    def generator(self, batch):
        grey_batch = rgb_to_grey(batch)
        with tf.variable_scope('g_') as vs:
            """ -----------------------------------------------------------------------------------
            ENCODER 
            ----------------------------------------------------------------------------------- """
            self.en_h0 = conv2d(grey_batch, self.frame_size, 128, k_h=4, k_w=4, d_w=2, d_h=2, name="enc_conv1")
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

            self.fg_h0 = tf.reshape(self.en_h3, [-1, 2, 4, 4, 512])
            print(self.fg_h0.get_shape().as_list())

            self.fg_h1 = conv3d_transpose(self.fg_h0, 512, [self.batch_size, 4, 8, 8, 256], name='g_f_h1')
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

            gen_reg = tf.reduce_mean(tf.square(grey_batch - rgb_to_grey(self.fg_fg)))

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

        self.input_images = tf.placeholder(tf.float32,
                                           [self.batch_size, self.frame_size, self.crop_size, self.crop_size, 3])
        self.videos_fake, self.gen_reg, self.generator_variables = self.generator(self.input_images)

        self.d_real, self.discriminator_variables = self.discriminator(self.input_images, reuse=False)
        self.d_fake, _ = self.discriminator(self.videos_fake, reuse=True)

        self.g_cost_pure = -tf.reduce_mean(self.d_fake)
        self.g_cost = self.g_cost_pure + 1000 * self.gen_reg

        self.d_cost = tf.reduce_mean(self.d_fake) - tf.reduce_mean(self.d_real)

        tf.summary.scalar("g_loss_pure", self.g_cost_pure)
        tf.summary.scalar("g_loss_regularizer", self.gen_reg)
        tf.summary.scalar("d_loss", self.d_cost)
        tf.summary.scalar("g_loss", self.g_cost)

        alpha = tf.random_uniform(
            shape=[self.batch_size, 1],
            minval=0.,
            maxval=1.
        )

        dim = self.frame_size * self.crop_size * self.crop_size * 3

        vid = tf.reshape(self.input_images, [self.batch_size, dim])
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
            self.g_adam = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.beta1, beta2=0.999) \
                .minimize(self.g_cost, var_list=self.generator_variables)

        print("\nTrainable variables for generator:")
        for var in self.generator_variables:
            print(var.name)
        print("\nTrainable variables for discriminator:")
        for var in self.discriminator_variables:
            print(var.name)

        self.sample = sampleBatch(self.videos_fake, self.batch_size)
        self.sample_ = sampleBatch_(rgb_to_grey(self.input_images), self.batch_size)
        self.summary_op = tf.summary.merge_all()

    def _train(self, loss_val, var_list, optimizer):
        grads = optimizer.compute_gradients(loss_val, var_list=var_list)
        for grad, var in grads:
            add_gradient_summary(grad, var)
        return optimizer.apply_gradients(grads)

    def get_feed_dict(self, session):
        images = session.run(self.videos)
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
        session.run(self.g_adam, feed_dict=feed_dict)

        if log_summary:
            g_loss_pure, g_reg, g_loss_total, d_loss_val, summary = session.run(
                [self.g_cost_pure, self.gen_reg, self.g_cost, self.d_cost, self.summary_op],
                feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            print("Time: %g/itr, Step: %d, generator loss: %g (%g + %g), discriminator_loss: %g" % (
                time.time() - start_time, step, g_loss_total, g_loss_pure, g_reg, d_loss_val))

        if generate_sample:
            vid_sample, bw_sample = session.run([self.sample, self.sample_], feed_dict=self.get_feed_dict(session))
            saveGIFBatch(vid_sample, sample_dir, 'vid_%d_col' % step)
            saveGIFBatch(bw_sample, sample_dir, 'vid_%d_bw' % step)


def add_activation_summary(var):
    tf.summary.histogram(var.op.name + "/activation", var)
    tf.summary.scalar(var.op.name + "/sparsity", tf.nn.zero_fraction(var))


def add_gradient_summary(grad, var):
    if grad is not None:
        tf.summary.histogram(var.op.name + '/gradient', grad)
