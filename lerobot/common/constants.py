# keys
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

OBS_ENV = "observation.environment_state"
OBS_ROBOT = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )


# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
QUAD_START_TOKEN = '<quad>'
QUAD_END_TOKEN = '</quad>'
REF_START_TOKEN = '<ref>'
REF_END_TOKEN = '</ref>'
BOX_START_TOKEN = '<box>'
BOX_END_TOKEN = '</box>'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.4814546, 0.4578275, 0.40821073)
CLIP_STD = (0.2686295, 0.2613025, 0.2757711)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

COMPRESS_SC_TOKEN = 'CP_SC'
COMPRESS_ACTION_TOKEN = 'CP_ACT'

ANSWER_LIST = [
    "Scene representations: [CP_SC]. Action representations: [CP_ACT].",
    "The results for the scene are [CP_SC], while the motion is represented as [CP_ACT].",
]

QUESTION_LIST = [
    "According to the instruction '{sent}', compress the video into scene-level and action-level representations.",
    "Observe the video frames under the task '{sent}', and summarize them into scene-level representations and action-level representations.",
]