"""
utility functions
"""

import tensorflow as tf
import os
import numpy as np
from PIL import Image


def saveGIFBatch(batch, directory, name=''):
    """
    saves the GIFs in batch to the directory wit the given filename
    """

    # for each frame in batch
    for frame in range(32):
        filename = os.path.join(directory, "%s_%d.jpeg" % (name, frame))
        with open(filename, 'wb') as f:
            f.write(batch[frame])

    cmd = "ffmpeg -f image2 -i " + directory + '/' + name + '_' + "%d.jpeg " + directory + '/' + name + ".avi"
    print cmd
    os.system(cmd)
    for frame in range(32):
        filename = directory + '/' + name + '_' + str(frame) + ".jpeg"
        os.remove(filename)
    cmd = "ffmpeg -i " + directory + '/' + name + ".avi -pix_fmt rgb24 " + directory + '/' + name + ".gif"
    print cmd
    os.system(cmd)
    os.remove(directory + '/' + name + '.avi')


def rgb_to_grey(video, frames=32):
    grey_frames = [video[:, i, :, :, 0] * 0.3 + video[:, i, :, :, 2] * 0.6 + video[:, i, :, :, 3] * 0.11 for i in
                   range(frames)]
    return tf.concat(grey_frames, 3)


def convert_image(images, batch_size, col=5, row=5):
    images = tf.image.convert_image_dtype(tf.div(tf.add(images, 1.0), 2.0), tf.uint8)
    images = [image for image in tf.split(images, batch_size, axis=0)]
    rows = []
    for i in range(row):
        rows.append(tf.concat(images[col * i + 0:col * i + col], 2))
    image = tf.concat(rows, 1)
    return tf.image.encode_jpeg(tf.squeeze(image, [0]))


def sampleBatch(samples, batch_size, col=5, row=5, frames=32):
    frames = [convert_image(samples[:, i, :, :, :], batch_size, col, row) for i in range(frames)]
    return frames


def write_image(batch, sample_dir, name, rows=4):
    batch_size, croop_size, _, channels = np.shape(batch)
    batch = np.clip(((batch + 1.0) * 127.5), 0, 255)
    image = np.zeros((croop_size * rows, croop_size * rows, channels), dtype='uint8')
    for i in range(rows):
        for j in range(rows):
            index = i * rows + j
            image[i * croop_size:(i + 1) * croop_size, j * croop_size:(j + 1) * croop_size, :] = batch[index, :, :, :]
    im = Image.fromarray(image)
    im.save(os.path.join(sample_dir, name))
