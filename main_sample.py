from model.improved_video_gan import ImprovedVideoGAN

import os
import tensorflow as tf
import numpy as np
from data.input_pipeline import InputPipeline

from PIL import Image

def write_batch(batch, sample_dir, name, it, rows, cols):
    batch = batch.astype('uint8')
    batch_size, frames, crop_size, _, _ = np.shape(batch)
    for i in range(frames):
        _write_image(batch, sample_dir, name, i, rows, cols)
    cmd = "ffmpeg -f image2 -i "+sample_dir +'/'+ name+'_'+"%d.png "+sample_dir+'/'+name+str(it)+".gif"
    print cmd
    os.system(cmd)
    for frame in range(frames):
        filename = os.path.join(sample_dir, "%s_%d.png" % (name, frame))
        os.remove(filename)


def _write_image(batch, sample_dir, name, frame, rows, cols):
    batch_size, _, croop_size,_, _ = np.shape(batch)
    image = np.zeros((croop_size * rows, croop_size * cols, 3), dtype='uint8')
    index = 0
    for i in range(rows):
        for j in range(cols):
            image[i * croop_size:(i + 1) * croop_size, j * croop_size:(j + 1) * croop_size, :] = batch[index, frame, :, :, :]
            index +=1
    im = Image.fromarray(np.asarray(np.clip(image, 0, 255), dtype="uint8"), "RGB")
    im.save(os.path.join(sample_dir, "%s_%d.png" % (name, frame)))

#
# Configuration for running on ETH GPU cluster
#
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SGE_GPU']

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

#
# input flags
#
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 128, 'Batch size [16]')
flags.DEFINE_integer('crop_size', 64, 'Crop size to shrink videos [64]')
flags.DEFINE_integer('frame_count', 32, 'How long videos should be in frames [32]')
flags.DEFINE_integer('z_dim', 100, 'Dimensionality of hidden features [100]')
flags.DEFINE_string('experiment_name', 'iWGAN_golf_advanced_newObjective', 'Log directory')
flags.DEFINE_string('checkpoint', 'cp-74600', 'checkpoint to recover')
flags.DEFINE_string('root_dir', '/srv/glusterfs/kratzwab/yt-bb-airplanes',
                    'Directory containing all videos and the index file')
flags.DEFINE_string('index_file', 'my-index-file.txt', 'Index file referencing all videos relative to root_dir')
params = flags.FLAGS

root_dir = '/scratch_net/boyscouts/kratzwab/experiments'
experiment_dir = os.path.join(root_dir, params.experiment_name)
checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
sample_dir = os.path.join(experiment_dir, 'generated-'+params.checkpoint)
if not os.path.exists(sample_dir):
    os.mkdir(sample_dir)

model = ImprovedVideoGAN(tf.random_uniform([params.batch_size, params.frame_count, params.crop_size, params.crop_size, 3]),
                  alpha1=0.1,
                  batch_size=params.batch_size,
                  frame_size=params.frame_count,
                  crop_size=params.crop_size,
                  learning_rate=0,
                  z_dim=params.z_dim,
                  beta1=0)
data_set = InputPipeline(params.root_dir,
                      params.index_file,
                      12,
                      params.batch_size,
                      num_epochs=10,
                      video_frames=params.frame_count,
                      reshape_size=params.crop_size)
batch = data_set.input_pipeline()

#
# Set up coordinator, session and thread queues
#
saver = tf.train.Saver()
coord = tf.train.Coordinator()

sess = tf.Session(config=config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
saver.restore(sess, os.path.join(checkpoint_dir,params.checkpoint))
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(5):
    batch_z = np.random.normal(0.0, 1.0, size=[params.batch_size, params.z_dim]).astype(np.float32)
    feed_dict = {model.z_vec: batch_z}
    x = sess.run(model.videos_fake, feed_dict=feed_dict)
    x = (x + 1)*127.5
    write_batch(x, sample_dir, 'test_', i, 16, 8)

#
# Shut everything down
#
coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
