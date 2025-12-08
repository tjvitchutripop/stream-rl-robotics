In addition to the files in this repository, make sure to clone the following projects into this directory:

Streaming-RL: https://github.com/mohmdelsayed/streaming-drl.git (note to use the training scripts in this repo you need to change the name of the directory from "streaming-drl" to "streaming_drl")

You will also need to install ManiSkill3: https://maniskill.readthedocs.io/en/latest/ (ideally do this from source)

To get closer to a working environment, create a conda environment based on the instructions on the Streaming-RL repo and then install the dependencies for ManiSkill3 afterwards. Once you've done that, try running the stream_ac_training_adapt.py script (but make sure do_damage is False) as there may still be a couple dependencies you need to install manually (e.g., wandb).