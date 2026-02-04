CUDA_VISIBLE_DEVICES=0 python stream_ac_training_adapt.py --damage_type slippery_floor
CUDA_VISIBLE_DEVICES=0 python stream_ac_training_adapt.py --damage_type broken_leg
CUDA_VISIBLE_DEVICES=0 python stream_ac_training_adapt.py --damage_type goal_shift
CUDA_VISIBLE_DEVICES=0 python stream_ac_training_adapt.py --damage_type slippery_floor --cbp
CUDA_VISIBLE_DEVICES=0 python stream_ac_training_adapt.py --damage_type broken_leg --cbp
CUDA_VISIBLE_DEVICES=0 python stream_ac_training_adapt.py --damage_type goal_shift --cbp