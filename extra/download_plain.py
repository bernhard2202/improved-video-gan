########################################################################
# YouTube BoundingBox Downloader
########################################################################
#
# This script downloads all videos within the YouTube BoundingBoxes
# dataset and cuts them to the defined clip size. It is accompanied by
# a second script which decodes the videos into single frames.
#
# Author: Mark Buckler
#
########################################################################
#
# The data is placed into the following directory structure:
#
# dl_dir/videos/d_set/class_id/clip_name.mp4
#
########################################################################



#
# Code heavily modified by Bernhard Kratzwald
#


from __future__ import unicode_literals
from subprocess import check_call
from concurrent import futures
import subprocess
import os
import sys
import csv

# The data sets to be downloaded
d_sets = ['yt_bb_classification_train',
          'yt_bb_classification_validation']

# Host location of segment lists
web_host = 'https://research.google.com/youtube-bb/'

global_dir = ''
glob_count = 1

MIN_MATCH_COUNT = 10
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


# Download and cut a clip to size
def dl_and_cut(vid):
    d_set_dir = vid.clips[0].d_set_dir
    # Use youtube_dl to download the video
    FNULL = open(os.devnull, 'w')
    try:
        check_call(['youtube-dl',
                    '--no-progress',
                    '-framerate', '25'
                                  '-f', 'best[ext=mp4]',
                    '-o', d_set_dir + '/' + vid.yt_id + '_temp.mp4',
                    'youtu.be/' + vid.yt_id],
                   stdout=FNULL, stderr=subprocess.STDOUT)
        sys.stderr.write('DONE...\n')
    except subprocess.CalledProcessError:
        sys.stderr.write("download failed: " + vid.yt_id)
        return

    # sys.stderr.write(str(len(vid.clips))+'\n')
    for clip in vid.clips:
        # Verify that the video has been downloaded. Skip otherwise
        if os.path.exists(d_set_dir + '/' + vid.yt_id + '_temp.mp4'):
            # Make the class directory if it doesn't exist yet
            class_dir = d_set_dir + '/' + vid.yt_id
            check_call(['mkdir', '-p', class_dir])
            sys.stderr.write(class_dir + '\n')
            # Cut out the clip within the downloaded video and save the clip
            # in the correct class directory. Note that the -ss argument coming
            # first tells ffmpeg to start off with an I frame (no frozen start)
            try:
                video = class_dir + '/' + clip.name + '.mp4'
                check_call(['ffmpeg',
                            '-ss', str(float(clip.start) / 1000),
                            '-i', 'file:' + d_set_dir + '/' + vid.yt_id + '_temp.mp4',
                            '-t', str((float(clip.start) + float(clip.stop)) / 1000),
                            '-c', 'copy', video],
                           stdout=FNULL, stderr=subprocess.STDOUT)
                open(global_dir + "job-list.txt", "a").write(video + '\n')
            except subprocess.CalledProcessError:
                continue
        else:
            sys.stderr.write("ERROR!!!")

    # Remove the temporary video
    try:
        os.remove(d_set_dir + '/' + vid.yt_id + '_temp.mp4')
    except OSError:
        return


# Parse the annotation csv file and schedule downloads and cuts
def parse_and_sched(dl_dir='videos', num_threads=4):
    """Download the entire youtube-bb data set into `dl_dir`.
    """

    # Make the download directory if it doesn't already exist
    check_call(['mkdir', '-p', dl_dir])

    open(global_dir + "job-list.txt", "w").write("\n")

    # For each of the four datasets
    for d_set in d_sets:

        # Make the directory for this dataset
        d_set_dir = dl_dir + 'frames'
        check_call(['mkdir', '-p', d_set_dir])

        # Download & extract the annotation list
        print (d_set + ': Downloading annotations...')
        # check_call(['wget', web_host + d_set + '.csv.gz'])
        # print (d_set + ': Unzipping annotations...')
        try:
            check_call(['gzip', '-d', '-f', d_set + '.csv.gz'])
        except:
            pass

        print (d_set + ': Parsing annotations into clip data...')
        # Parse csv data
        with open(('../' + d_set + '.csv'), 'rt') as f:
            reader = csv.reader(f)
            annotations = list(reader)

        # Sort to de-interleave the annotations for easier parsing
        if ('classification' in d_set):
            class_or_det = 'class'
            # Sort by youtube_id, class, and then timestamp
            annotations.sort(key=lambda l: (l[0], l[2], l[1]))
        elif ('detection' in d_set):
            class_or_det = 'det'
            # Sort by youtube_id, class, obj_id and then timestamp
            annotations.sort(key=lambda l: (l[0], l[2], l[4], l[1]))

        current_clip_name = ['blank']
        clips = []

        # Parse annotations into list of clips with names, youtube ids, start
        # times and stop times
        for idx, annotation in enumerate(annotations):
            # print(idx)
            # If this is for a classify dataset there is no object id
            if (class_or_det == 'class'):
                obj_id = '0'
            elif (class_or_det == 'det'):
                obj_id = annotation[4]
            yt_id = annotation[0]
            class_id = annotation[2]
            # print(class_id)

            clip_name = yt_id + '+' + class_id + '+' + obj_id

            # If this is a new clip
            if clip_name != current_clip_name:

                # Update the finishing clip
                if idx != 0:  # If this isnt the first clip
                    clips[-1].stop = annotations[idx - 1][1]

                # Add the starting clip
                clip_start = annotation[1]
                clips.append(video_clip(
                    clip_name,
                    yt_id,
                    clip_start,
                    '0',
                    class_id,
                    obj_id,
                    d_set_dir))

                # Update the current clip name
                current_clip_name = clip_name

        # Update the final clip with its stop time
        clips[-1].stop = annotations[-1][1]

        # Sort the clips by youtube id
        clips.sort(key=lambda x: x.yt_id)
        print('bla')
        # Create list of videos to download (possibility of multiple clips
        # from one video)
        current_vid_id = ['blank']
        vids = []
        for clip in clips:
            # USE THIS TO FILER FOR A CLASS:
            if int(clip.class_id) != 7:
                continue
            if os.path.exists(clip.d_set_dir + '/' + clip.yt_id):
                print("SKIP {}".format(clip.d_set_dir))
                continue

            vid_id = clip.yt_id

            # If this is a new video
            if vid_id != current_vid_id:
                # Add the new video with its first clip
                vids.append(video(clip.yt_id, clip))
            # If this is a new clip for the same video
            else:
                # Add the new clip to the video
                vids[-1].clips.append(clip)

            # Update the current video name
            current_vid_id = vid_id

        # Download and cut in parallel threads giving
        with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            fs = [executor.submit(dl_and_cut, vid) for vid in vids]
            for i, f in enumerate(futures.as_completed(fs)):
                # Write progress to error so that it can be seen
                sys.stderr.write("Downloaded video: {} / {} \n".format(i, len(vids)))


if __name__ == '__main__':
    # Use the directory `videos` in the current working directory by
    # default, or a directory specified on the command line.
    parse_and_sched(global_dir, 3)
