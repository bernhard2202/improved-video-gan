#!/bin/bash
#
## otherwise the default shell would be used
#$ -S /bin/bash
#
## Use GPU
#$ -l gpu
#
## <= 2h is short queue, <= 24h is middle queue, <= 120 h is long queue
#$ -l h_rt=24:00:00
#
## the maximum memory usage of this job, (below 4G does not make much sense)
#$ -l h_vmem=70G
#
## stderr and stdout are merged together to stdout
#$ -j y
#
# logging directory. preferrably on your scratch
#$ -o /scratch_net/boyscouts/kratzwab/experiments

# if you need to export custom libs, you can do that here

export SGE_GPU_ALL="$(ls -rt /tmp/lock-gpu*/info.txt | xargs grep -h  $(whoami) | awk '{print $2}' | paste -sd "," -)"
export SGE_GPU=$(echo $SGE_GPU_ALL |rev|cut -d, -f1|rev) # USE LAST GPU by request time
echo "SGE gpu=$SGE_GPU_ALL available"
echo "SGE gpu=$SGE_GPU allocated in this use"


## python /scratch_net/boyscouts/kratzwab/workspace/improved-video-gan/main_train.py --experiment_name=iWGAN_future_alternating --mode=predict --recover_model=True --batch_size=64 --crop_size=64 --learning_rate=0.0002 --root_dir=/srv/glusterfs/kratzwab/golf/frames-stable-many --index_file=golf.txt
## python /scratch_net/boyscouts/kratzwab/workspace/improved-video-gan/main_train.py --experiment_name=iWGAN_airplanes_shot_advanced_newObjective --mode=generate --recover_model=True --batch_size=64 --crop_size=64 --learning_rate=0.0002  --use_two_stream=False  --root_dir=/srv/glusterfs/kratzwab/yt-bb-airplanes-shot --index_file=index-file1.txt
python /scratch_net/boyscouts/kratzwab/workspace/improved-video-gan/main_train.py --experiment_name=iWGAN_inpaint --mode=inpaint --recover_model=True --batch_size=32 --crop_size=64 --learning_rate=0.0002 --root_dir=/srv/glusterfs/kratzwab/golf/frames-stable-many --index_file=golf.txt
