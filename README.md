# An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics

Training and evaluation code for the experiments in *An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics*. The repository compares streaming actor–critic adaptation with batch PPO baselines under changes to robot dynamics and tasks.

The experiments use [ManiSkill](https://maniskill.readthedocs.io/) environments and include custom task variants for quadruped reach and PushCube.

## What is included

- Streaming actor–critic adaptation with `AdaptiveObGD`, `ObGD`, or Adam.
- Batch PPO pre-training and an adaptation baseline.
- Quadruped reach (`AnymalC-Reach-v1`), manipulation (`PushCube-v1`), and humanoid transport (`UnitreeG1TransportBox-v1`) experiment scripts.
- Perturbations including goal shifts, friction changes, joint/leg damage, and humanoid arm-and-hand impairment.
- Pre-trained PPO checkpoints in `pretrained-models/`.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- A working ManiSkill installation. GPU acceleration is strongly recommended for the vectorized PPO pre-training workflow.

## Setup

Clone this repository, create the locked project environment, then install ManiSkill into that environment:

```bash
git clone https://github.com/tjvitchutripop/stream-rl-robotics.git
cd stream-rl-robotics
uv sync
git clone https://github.com/haosulab/ManiSkill.git
uv pip install -e ./ManiSkill
```

All examples below run through `uv`, so manually activating `.venv` is unnecessary. If you prefer to invoke `python` directly, activate it first:

```bash
source .venv/bin/activate
```

## Run an experiment

Each adaptation script accepts command-line options. Run `--help` to see the full set for a given script.

### Quadruped reach

Start streaming adaptation from the included AnymalC PPO checkpoint. This default configuration introduces the selected perturbation after 500,000 environment steps.

```bash
uv run stream_ac_training_adapt_quad.py \
  --checkpoint pretrained-models/anymalc-reach/adam_ppo_pretrain.pt \
  --damage_type slippery_floor
```

Use the Adam-specific variant for an Adam optimizer baseline:

```bash
uv run stream_ac_training_adapt_quad_adam.py \
  --checkpoint pretrained-models/anymalc-reach/adam_ppo_pretrain.pt \
  --damage_type broken_leg
```

The paper uses the `goal_shift`, `broken_leg`, and `slippery_floor` conditions. Easier variants, including `stuck_joint`, `slippery_floor_easy`, and `goal_shift_easy`, are also available in the code but were not used in the paper.

### PushCube manipulation

```bash
uv run stream_ac_training_adapt_manip.py \
  --checkpoint pretrained-models/push-cube/adam_ppo_pretrain.pt \
  --damage_type goal_shift
```

`custom_envs/push_cube.py` registers `PushCube-v1` and implements task changes such as goal shifts and cube/table friction changes.
For the manipulation `goal_shift` condition, the goal is offset by `-0.15` on the y-axis.

### Humanoid transport

```bash
uv run stream_ac_training_adapt_humanoid.py \
  --checkpoint pretrained-models/transport-box/adam_ppo_pretrain.pt \
  --do_damage \
  --damage_type arm_hand_impairment
```

The humanoid script defaults to `UnitreeG1TransportBox-v1`. Its arm/hand impairment is configurable with `--arm_shoulder_elbow_gain`, `--arm_finger_gain`, and `--arm_stuck_joint_idxs`.

### PPO pre-training and batch adaptation

To train a PPO policy from scratch, use the vectorized pre-training script (its defaults target `AnymalC-Reach-v1`):

```bash
uv run batch_ppo_pretrain.py --env-id AnymalC-Reach-v1
```

To run the batch PPO adaptation baseline:

```bash
uv run batch_ppo_adapt_quad.py \
  --checkpoint pretrained-models/anymalc-reach/adam_ppo_pretrain.pt \
  --damage_type broken_leg
```

## Checkpoints

The repository ships pre-trained PPO weights for the three main tasks:

| Task | Checkpoint directory |
| --- | --- |
| Quadruped reach | `pretrained-models/anymalc-reach/` |
| PushCube manipulation | `pretrained-models/push-cube/` |
| Humanoid transport | `pretrained-models/transport-box/` |

Both standard and layer-normalized (`*_ln.pt`) checkpoints are provided. Pass a checkpoint path with `--checkpoint` when running an adaptation script. When loading a layer-normalized checkpoint, also pass `--layernorm`.

## Evaluate a training checkpoint

`evaluate_policy.py` evaluates a compatible checkpoint saved by a training run and reports mean return and success rate. It uses deterministic policy means by default; add `--stochastic` to sample actions. Damage is opt-in with `--damage-type`.

```bash
uv run evaluate_policy.py \
  --task quad \
  --checkpoint weights/stream_ac_AnymalC-Reach-v1_<timestamp>/seed_0.pth \
  --damage-type broken_leg \
  --episodes 50
```

Use `--task manip` or `--task humanoid` for the other tasks. The `--help` output lists each task's supported optional damage conditions and parameters. In particular, `--task manip --damage-type goal_shift` uses the paper's `-0.15` y-axis offset unless overridden with `--goal-offset-y`.

## Outputs and tracking

Adaptation runs write artifacts under timestamped directories:

- `logs/` — training and evaluation logs
- `results/` — serialized returns and termination-step data
- `weights/` — final model weights; the humanoid workflow also saves the best post-perturbation checkpoint

Weights & Biases logging is enabled by default in the adaptation scripts. Authenticate with `uv run wandb login`, or modify the script/configuration for offline or disabled logging before launching a run.

## Repository layout

```text
custom_envs/                    Custom ManiSkill environments and perturbation hooks
pretrained-models/              Included PPO initialization checkpoints
streaming_drl/                  Optimizers, initialization, normalization, and time wrappers
batch_ppo_pretrain.py           Vectorized PPO pre-training
batch_ppo_adapt_quad.py         Batch PPO quadruped adaptation baseline
stream_ac_training_adapt_*.py   Streaming adaptation entry points
model.py                        Actor and critic architectures
interpretability.py             Optional Weights & Biases MLP diagnostics
```
