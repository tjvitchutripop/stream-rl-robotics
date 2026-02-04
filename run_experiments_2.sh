CUDA_VISIBLE_DEVICES=1 python stream_ac_training_adapt.py --damage_type slippery_floor_easy
CUDA_VISIBLE_DEVICES=1 python stream_ac_training_adapt.py --damage_type stuck_joint
CUDA_VISIBLE_DEVICES=1 python stream_ac_training_adapt.py --damage_type goal_shift_easy
CUDA_VISIBLE_DEVICES=1 python stream_ac_training_adapt.py --damage_type stuck_joint --cbp
CUDA_VISIBLE_DEVICES=1 python stream_ac_training_adapt.py --damage_type goal_shift_easy --cbp
CUDA_VISIBLE_DEVICES=1 python stream_ac_training_adapt.py --damage_type slippery_floor_easy --cbp