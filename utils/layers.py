import tensorflow as tf
import numpy as np


def conv2d(input_, input_dim, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, name="conv2d", padding="SAME"):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    """ 
    init weights like in    
    "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
    """
    with tf.variable_scope(name):
        fan_in = input_dim * k_h * k_w
        fan_out = (output_dim * k_h * k_w) / (d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_h, k_w, input_dim, output_dim)
        )

        w_init = tf.Variable(filter_values, name='filters_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        result = tf.nn.conv2d(
            input=input_,
            filter=w,
            strides=[1, d_h, d_w, 1],
            padding=padding,
            data_format='NHWC'
        )
        result = tf.nn.bias_add(result, b)

    return result


def conv3d(input_, input_dim, output_dim,
           k_t=4, k_h=4, k_w=4, d_t=2, d_h=2, d_w=2, name="conv3d", padding="SAME"):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    with tf.variable_scope(name):
        """ 
        init weights like in 
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        """
        fan_in = input_dim * k_t * k_h * k_w
        fan_out = (output_dim * k_t * k_h * k_w) / (d_t * d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_t, k_h, k_w, input_dim, output_dim)
        )

        w_init = tf.Variable(filter_values, name='filters_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        result = tf.nn.conv3d(
            input=input_,
            filter=w,
            strides=[1, d_t, d_h, d_w, 1],
            padding=padding,
            data_format='NDHWC'
        )
        result = tf.nn.bias_add(result, b)

    return result


def conv3d_transpose(input_, input_dim, output_shape,
                     k_h=4, k_w=4, k_d=4, d_h=2, d_w=2, d_d=2,
                     name="deconv3d"):
    def uniform(std_dev, size):
        return np.random.uniform(
            low=-std_dev * np.sqrt(3),
            high=std_dev * np.sqrt(3),
            size=size
        ).astype('float32')

    with tf.variable_scope(name):
        """ 
        init weights like in 
        "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
        """

        output_dim = output_shape[-1]

        fan_in = input_dim * k_d * k_h * k_w
        fan_out = (output_dim * k_d * k_h * k_w) / (d_d * d_h * d_w)

        filters_std = np.sqrt(4. / (fan_in + fan_out))

        filter_values = uniform(
            filters_std,
            (k_d, k_h, k_w, output_dim, input_dim)
        )

        w_init = tf.Variable(filter_values, name='filter_init')
        w = tf.get_variable('filters', initializer=w_init.initialized_value())
        b = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))

        result = tf.nn.conv3d_transpose(value=input_,
                                        filter=w,
                                        output_shape=output_shape,
                                        strides=[1, d_d, d_h, d_w, 1],
                                        name=name,
                                        )

        result = tf.nn.bias_add(result, b)
        return result


def leaky_relu(x, leak=0.2):
    """
    Code taken from https://github.com/sugyan/tf-dcgan/blob/master/dcgan.py
    """
    return tf.maximum(x, x * leak)


def dis_block(input, input_dim, output_dim, name, reuse=False, normalize=True):
    with tf.variable_scope(name, reuse=reuse) as vs:
        result = conv3d(input, input_dim, output_dim, name='conv3d')
        if normalize:
            result = tf.contrib.layers.layer_norm(result, reuse=reuse, scope=vs)
        result = leaky_relu(result)
    return result


def linear(input_, output_size, scope=None, stddev=0.01, bias_start=0.0, with_w=False):
    """
    Code from https://github.com/wxh1996/VideoGAN-tensorflow
    """
    shape = input_.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
                               initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
