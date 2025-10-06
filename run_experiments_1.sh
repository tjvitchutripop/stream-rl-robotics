# CUDA_VISIBLE_DEVICES=0 python batch_training_adapt.py --damage_type broken_leg
CUDA_VISIBLE_DEVICES=0 python batch_training_adapt.py --damage_type slippery_floor
CUDA_VISIBLE_DEVICES=0 python batch_training_adapt.py --damage_type goal_shift
CUDA_VISIBLE_DEVICES=0 python ppo_stream_pretrain_adam.py