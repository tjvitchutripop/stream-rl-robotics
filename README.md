# Stream RL Robotics

This is a repository containing the training and evaluation scripts for the paper "An Analysis of Streaming Deep Reinforcement Learning for Adaptive Continual Learning in Robotics".

## Installation & Setup

Follow these steps to set up the environment and install dependencies.

### 1. Install UV and Sync Environment

First, make sure you have `uv` installed. If not, follow the official [installation guide](https://github.com/astral-sh/uv).

Once `uv` is installed, sync the project environment:

```bash
uv sync
```

### 2. Install ManiSkill

Clone the ManiSkill repository and install it locally using `uv`:

```bash
git clone https://github.com/haosulab/ManiSkill.git
cd ManiSkill
uv pip install -e .
cd ..
```

## Source virtual environment
You don't need to do this if you are using `uv run` but if that doesn't work for you, do this before running the training scripts.

```bash
source .venv/bin/activate
```

## Running Experiments

You can now run the training scripts using `uv run` or `python`. For example:

```bash
uv run stream_ac_training.py
```
or 
```bash
python stream_ac_training.py
```
