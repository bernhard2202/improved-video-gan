Towards an Understanding of Our World by GANing Videos in the Wild
==================================================================

GitHubb repository for "Towards an Understanding of Our World by GANing Videos in the Wild" 
[Paper Link](https://github.com/bernhard2202/improved-video-gan/raw/master/paper/paper.pdf)


Requirements
------------
* Tensorflow 1.2.1
* Python 2.7
* ffmpeg

Data
----
Videos are stored as JPEGs of vertically stacked frames. Every frame needs to be at least 64x64 pixels and contains between 16 and 32 frames. 
For example datasets see: http://carlvondrick.com/tinyvideo/#data


Training
--------

python main_train.py 

Important Parameters:

* mode: one of 'generate', 'predict', 'bw2rgb', 'inpaint' depending on weather you want to generate videos, predict future frames, colorize videos or do inpainting.
* batch_size: Recommended 64, for colorization use 32 for memory issues. 
* root_dir: root directory of dataset
* index_file: must be in root_dir, containing a list of all training data clips; path relative to root_dir.
* experiment_name: name of experiment
* output_every: output loss to stdout and write to tensorboard summary every xx steps.
* sample_every: generate a visual sample every xx steps.
* save_model_very: save the model every xx steps.
* recover_model: if true recover model and continue training


