"""
Original from: https://github.com/cvondrick/videogan/blob/master/extra/stabilize_videos_many.py

Heavily modified by Bernhard Kratzwald.

"""

from __future__ import unicode_literals
from subprocess import check_call
import subprocess
import os
import numpy as np
import cv2

import random

global_dir = '/'
glob_count = 1

MIN_MATCH_COUNT = 4
VIDEO_SIZE = 128
CROP_SIZE = 128
MAX_FRAMES = 33
MIN_FRAMES = 16
FRAMES_DELAY = 2


# Video clip class
class video_clip(object):
    def __init__(self, name, yt_id, start, stop, class_id, obj_id, d_set_dir):
        # name = yt_id+class_id+object_id
        self.name = name
        self.yt_id = yt_id
        self.start = start
        self.stop = stop
        self.class_id = class_id
        self.obj_id = obj_id
        self.d_set_dir = d_set_dir

    def print_all(self):
        print('[' + self.name + ', ' +
              self.yt_id + ', ' +
              self.start + ', ' +
              self.stop + ', ' +
              self.class_id + ', ' +
              self.obj_id + ']\n')


class video(object):
    def __init__(self, yt_id, first_clip):
        self.yt_id = yt_id
        self.clips = [first_clip]

    def print_all(self):
        print(self.yt_id)
        for clip in self.clips:
            clip.print_all()


# get info of video
def get_video_info(video):
    stats = subprocess.check_output(
        "ffprobe -select_streams v -v error -show_entries stream=width,height,duration -of default=noprint_wrappers=1 {}".format(
            video), shell=True)
    info = dict(x.split("=") for x in stats.strip().split("\n"))
    print info
    return {"width": int(info['width']),
            "height": int(info['height']),
            "duration": float(info['duration'])}


class FrameReader(object):
    def __init__(self, video):
        self.info = get_video_info(video)

        command = ["ffmpeg",
                   '-i', video,
                   '-f', 'image2pipe',
                   '-pix_fmt', 'rgb24',
                   '-vf', 'fps=25',
                   '-vcodec', 'rawvideo',
                   '-']
        self.pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10 ** 8)

    def __iter__(self):
        return self

    def next(self):
        raw_image = self.pipe.stdout.read(self.info['width'] * self.info['height'] * 3)
        # transform the byte read into a numpy array
        image = np.fromstring(raw_image, dtype='uint8')
        try:
            image = image.reshape((self.info['height'], self.info['width'], 3))
        except:
            raise StopIteration()
        # throw away the data in the pipe's buffer.
        self.pipe.stdout.flush()

        image = image[:, :, (2, 1, 0)]

        return image

    def close(self):
        self.pipe.stdout.close()
        self.pipe.kill()


def process_im(im):
    h = im.shape[0]
    w = im.shape[1]

    if w > h:
        scale = float(VIDEO_SIZE) / h
    else:
        scale = float(VIDEO_SIZE) / w

    new_h = int(h * scale)
    new_w = int(w * scale)

    im = cv2.resize(im, (new_w, new_h))

    h = im.shape[0]
    w = im.shape[1]

    h_start = h / 2 - CROP_SIZE / 2
    h_stop = h_start + CROP_SIZE

    w_start = w / 2 - CROP_SIZE / 2
    w_stop = w_start + CROP_SIZE

    im = im[h_start:h_stop, w_start:w_stop, :]
    return im


def split(video, dir):
    try:
        frames = FrameReader(video)
    except subprocess.CalledProcessError:
        print "failed due to CalledProcessError"
        return False

    for _ in range(FRAMES_DELAY):
        try:
            frames.next()
        except StopIteration:
            return False

    check_call(['mkdir', '-p', dir])

    movie_clip = 0
    movie_clip_files = []
    for _ in range(250):
        try:
            img2 = frames.next()
        except StopIteration:
            print "END OF STREAM"
            break

        failed = False
        bg_img = process_im(img2)
        movie = [bg_img.copy()]

        for _ in range(MAX_FRAMES):
            try:
                img2 = frames.next()
                movie.append(process_im(img2))
            except StopIteration:
                print "END OF STREAM"
                failed = True
                break

        if len(movie) < MIN_FRAMES:
            print "this movie clip is too short, causing fail"
            failed = True
        if failed:
            print ("aborting movie clip due to failure")
        else:
            # write a column stacked image so it can be loaded at once, which
            # will hopefully reduce IO significantly
            stacked = np.vstack(movie)
            movie_clip_filename = dir + "/%04d.jpg" % movie_clip
            print "writing {}".format(movie_clip_filename)
            movie_clip_files.append(movie_clip_filename)
            cv2.imwrite(movie_clip_filename, stacked)
            movie_clip += 1
    frames.close()
    print ("Written {} files.".format(len(movie_clip_files)))


def get_stable_path(path):
    if path.endswith(".mp4"):
        return path.replace(".mp4", "")
    else:
        return ""


work = [x.strip() for x in open(global_dir + "job-list.txt")]
random.shuffle(work)

i = 0
for video in work:
    stable_path = get_stable_path(video)
    lock_file = stable_path + ".lock"

    if stable_path == "":
        continue

    if os.path.exists(stable_path) or os.path.exists(lock_file):
        print "already done: {}".format(stable_path)
        i += 1
        continue

    try:
        os.makedirs(os.path.dirname(stable_path))
    except OSError:
        pass
    try:
        os.makedirs(stable_path)
    except OSError:
        pass
    try:
        os.mkdir(lock_file)
    except OSError:
        pass

    print video

    # result = compute("videos/" + video, stable_path)
    split(video, stable_path)

    try:
        os.remove(video)
    except:
        pass

    try:
        os.rmdir(lock_file)
    except:
        pass
