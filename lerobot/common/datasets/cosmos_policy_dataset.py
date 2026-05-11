#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import contextlib
import logging
import random
import json
import copy
import os
import shutil
import polars as pl
from pathlib import Path
from typing import Callable, Dict
from datetime import datetime

import datasets
import numpy as np
import packaging.version
import PIL.Image as Image
import torch
import torch.utils

from torch.utils.data import ConcatDataset, Subset
from torch.utils.data.dataloader import default_collate
import torch.nn.functional as F
from torchvision import transforms as T

import transformers
from transformers import Qwen2_5_VLProcessor, AutoTokenizer, InternVLProcessor
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
from qwen_vl_utils import process_vision_info

from datasets import concatenate_datasets, load_dataset, Dataset

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.constants import HF_LEROBOT_HOME, OBS_ROBOT
from lerobot.common.datasets.oxe_configs import OXE_DATASET_CONFIGS
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
from lerobot.common.datasets.utils import cycle, save_to_json
# from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.common.datasets.data_utils import preprocess_image
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    encode_video_frames,
    decode_video_frames,
    get_video_info,
)
from lerobot.common.robot_devices.robots.utils import Robot

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from tabulate import tabulate
import hashlib


CODEBASE_VERSION = "v2.1"
PAD_VALUE = {"attention_mask": 0, "input_ids": 151643, "labels": IGNORE_TOKEN_ID}

def duplicate_array(arr, total_num_copies):
    """
    Duplicates a NumPy array multiple times along a new first axis.

    Args:
        arr (numpy.ndarray): The input array to duplicate
        total_num_copies (int): Total number of copies to have in the end

    Returns:
        numpy.ndarray: A new array with shape (total_num_copies, *arr.shape)
    """
    # Create a new array by stacking the original array multiple times
    return np.stack([arr] * total_num_copies)

def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to: {seed}")


class NamedSubset(Subset):
    def __init__(self, dataset, indices, dataset_name):
        super().__init__(dataset, indices)
        self.dataset_name = dataset_name  # 存储数据集名称

    def __getitem__(self, idx):
        data = super().__getitem__(idx)  # 获取原始数据
        return {**data, "dataset_name": self.dataset_name}

def parquet_to_dataset(parquet_file: str, split: str = "train") -> datasets.Dataset:
    """Converts a parquet file to a Hugging Face dataset."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start reading parquet file: {parquet_file}")
    parquet_pl = pl.read_parquet(parquet_file)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start converting parquet to list")
    parquet_list = parquet_pl.to_dicts()
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Start creating dataset")
    dataset = Dataset.from_list(parquet_list, split=split)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Dataset created successfully.")
    return dataset

class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()
            
    def restrict_image_features(self, features: dict[str, dict], max_feature=8) -> dict[str, dict]:
        """Restricts the number of image features to a maximum number."""
        image_features = {k: v for k, v in features.items() if v["dtype"] in ["image", "video"]}
        if len(image_features) > max_feature:
            logging.warning(
                f"Found {len(image_features)} image features, restricting to {max_feature}."
            )
            num_features = len(image_features)
            image_features = dict(list(image_features.items())[:num_features-max_feature])
        # remove feature not in image features
        feature_to_return = features.copy()
        if len(image_features) > max_feature:
            for k in features.keys():
                if k in image_features.keys():
                    feature_to_return.pop(k)
        return feature_to_return
    def load_metadata(self):
        self.info = load_info(self.root)
        self.info['features'] = self.restrict_image_features(self.info['features'])
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        self.stats = load_stats(self.root)
        if self.stats == None:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))
        # if self._version < packaging.version.parse("v2.1"):
        #     self.stats = load_stats(self.root)
        #     self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        # else:
        #     self.episodes_stats = load_episodes_stats(self.root)
        #     self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
                )
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            # TODO(aliberts, rcadene): implement sanity check for features
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj
    
    @classmethod
    def create_with_stats_feats(
        cls, 
        stats, 
        features,
        fps = 30,
        robot_type = "all",
        use_videos = True,
        ) -> "LeRobotDatasetMetadata":
        obj = cls.__new__(cls)
        obj.stats = stats
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        wrist_image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        keep_img_keys: str | None = None,
        dataset_name: str = "default",
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. There is currently
                a single option which is the pyav decoder used by Torchvision. Defaults to pyav.
        """
        super().__init__()
        # print("__init__ 方法被调用")
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.wrist_image_transforms = wrist_image_transforms
        print(self.image_transforms, self.wrist_image_transforms)
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else "pyav"
        self.delta_indices = None
        self.keep_img_keys = keep_img_keys
        self.dataset_name = dataset_name

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )
        # print(f"Episodes in the dataset: {episodes}")
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Loading episodes stats...")
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Trying to load dataset {self.repo_id}...")
        try:
            if force_cache_sync:
                raise FileNotFoundError
            # assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
            
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Dataset loaded successfully, loading timestamps.")

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(list(self.hf_dataset["timestamp"])).numpy()
        episode_indices = torch.stack(list(self.hf_dataset["episode_index"])).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Checking timestamps sync status...")
        
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            # path = str(self.root / "data")
            path = str(self.root / "merged.parquet")
            # hf_dataset = parquet_to_dataset(parquet_file=path, split="train")
            hf_dataset = load_dataset("parquet", data_files=path, split="train")
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - Dataset length is {len(hf_dataset)}")
            # hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def expand_true(self, mask, k=2):
        mask = mask.clone()
        true_idx = mask.nonzero(as_tuple=True)[0]
        if len(true_idx) > 0:
            start = true_idx[0].item()
            new_start = max(0, start - k)
            mask[new_start:] = True   # 注意这里是从 new_start 到最后都置为 True
        return mask

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        # delta_indices:{"action" : [1, 2, 3, 4, 5]}
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        # query_indices["observation.images.image"] = query_indices["action"]
        # print(query_indices)
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]
        # for key, timestamps in query_timestamps.items():
        #     print(key, timestamps)
        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(list(self.hf_dataset.select(q_idx)[key]))
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            # frames = decode_video_frames_torchvision(
            #     video_path, query_ts, self.tolerance_s, self.video_backend
            # )
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend, return_type="numpy")
            # print(vid_key, frames.shape)
            item[vid_key] = frames

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames
    
    def resize_with_pad(self, img, width, height, pad_value=-1):
        # assume no-op when width height fits already
        need_expand = False
        if img.ndim != 4:
            need_expand = True
            img = img.unsqueeze(1)
            # raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

        cur_height, cur_width = img.shape[2:]

        ratio = max(cur_width / width, cur_height / height)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        resized_img = F.interpolate(
            img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
        )

        pad_height = max(0, int(height - resized_height))
        pad_width = max(0, int(width - resized_width))

        # pad on left and top of image
        padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
        if need_expand:
            padded_img = padded_img.squeeze(1)
        return padded_img
    

    def __getitem__(self, idx) -> dict:
        # print(f"Idx:{idx}")
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
            
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        
        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]
        item["dataset_name"] = self.dataset_name

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        img_dir = self.root / "images"
        if img_dir.is_dir():
            shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else "pyav"
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else {repo_id: 1e-4 for repo_id in repo_ids}
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps,
                tolerance_s=self.tolerances_s[repo_id],
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]

        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        # TODO(rcadene, aliberts): We should not perform this aggregation for datasets
        # with multiple robots of different ranges. Instead we should have one normalization
        # per robot.
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])

    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        item["dataset_index"] = torch.tensor(dataset_idx)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]

        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )


class MultiDatasetforDistTraining(torch.utils.data.Dataset):
    def __init__(self, cfg, data_mix, vla2root_json, seed, image_transforms = None, wrist_image_transforms = None):
        super().__init__()
        self.seed = seed
        self.stage = cfg.stage
        self.cfg = cfg
        # 1. prepare mixture dataset
        data_mixture = OXE_NAMED_MIXTURES[data_mix]
        included_d_names = []
        dataset_sampling_weights = []
        for d_name, d_weight in data_mixture:
            if d_name in included_d_names:
                print(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
                continue

            included_d_names.append(d_name)
            dataset_sampling_weights.append(d_weight)
        
        # make dataset
        self.datasets = []
        self.dataset_sizes = []
        self.dataset_names = []
        self.num_episodes = 0
        self.num_frames = 0
        parent_dir = cfg.dataset.parent_dir
        with open(vla2root_json, "r") as f:
            vla2data_root = json.load(f)
        for dataset_name in included_d_names:
            if dataset_name in vla2data_root.keys():
                data_root = vla2data_root[dataset_name]
                data_root = os.path.join(parent_dir, data_root)
                print(f"Load data from {data_root}")
                repo_id = f"bulldog-{dataset_name}" # any
                ds_meta = LeRobotDatasetMetadata(repo_id, root=data_root)
                delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
                dataset = LeRobotDataset(
                    repo_id, 
                    root=data_root,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms,
                    wrist_image_transforms=wrist_image_transforms,
                    video_backend=cfg.dataset.video_backend,
                    dataset_name=dataset_name,
                )
                self.num_episodes += dataset.num_episodes
                self.num_frames += dataset.num_frames
                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))
                self.dataset_names.append(dataset_name)
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {dataset_name} not found in vla2root.json, skipping...")

        # 2. Set properties for sampling
        self.set_epoch(0)
        self.balance_dataset_weights = cfg.dataset.balance_dataset_weights
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        
        print(f"Dataset lengths: {self._dataset_lengths} Num episodes:{self.num_episodes}")

        # Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights)
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        
        # Normalize weights
        weights_sum = self._dataset_sampling_weights.sum()
        if weights_sum == 0 or np.isnan(weights_sum):
            print(f"Error: Invalid weights sum: {weights_sum}")
            # Fallback to equal weights
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(self.datasets)
            print(f"Fallback to equal weights")
        else:
            self._dataset_sampling_weights /= weights_sum
        
        table_data = [
            [self.dataset_names[i], len(self.datasets[i]), f"{self._dataset_sampling_weights[i]:.4f}"]
                for i in range(len(self.datasets))
        ]
        print(tabulate(table_data, headers=["Dataset", "Frames", "Ratio"], tablefmt="grid"))
        print(f"Total frames: {self._dataset_lengths.sum()}")
        
        if self.stage == "pretrain":
            # 4. prepare dataset indicies for sampling
            self._step_order: list[np.ndarray] = []
            self._step_pos: list[int] = []
            for dataset in self.datasets:
                self._step_order.append(np.arange(len(dataset)))
                rng = np.random.default_rng(self.seed)
                rng.shuffle(self._step_order[-1])
                self._step_pos.append(0)
            self.dataset_len = np.max(self._dataset_lengths)
        else:
            self.full_dataset = ConcatDataset(self.datasets)
            self.dataset_len = len(self.full_dataset)
        
        # 4. Aggregate dataset stats from all datasets
        self.stats = aggregate_stats([dataset.meta.stats for dataset in self.datasets], 
                                     max_dim = cfg.policy.max_action_dim)
        
        # in fact, we do not use it, so just simply copy
        self.meta = ds_meta
        
        # other property
        self.use_proprio = cfg.policy.use_proprio
        self.use_wrist_images = cfg.policy.use_wrist_images
        self.use_third_person_images = cfg.policy.use_third_person_images
        self.num_duplicates_per_image = cfg.policy.num_duplicates_per_image
        self.final_image_size = cfg.policy.final_image_size
        self.normalize_images = cfg.policy.normalize_images
        self.use_image_aug = cfg.policy.use_image_aug
        self.use_stronger_image_aug = cfg.policy.use_stronger_image_aug
        self.max_action_dim = cfg.policy.max_action_dim
        self.max_state_dim = cfg.policy.max_state_dim
        
        
    def pad_vector(self, vector, new_dim):
        """Can be (batch_size x sequence_length x features_dimension)
        or (batch_size x features_dimension)
        """
        if vector.shape[-1] == new_dim:
            return vector
        shape = list(vector.shape)
        current_dim = shape[-1]
        shape[-1] = new_dim
        new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
        new_vector[..., :current_dim] = vector
        return new_vector
    
    def __len__(self):
        # return len(self.dataset)
        return self.dataset_len

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch
    
    def sample_step(self, index: int):
        seed = safe_hash((self.epoch, index, self.seed))
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self._dataset_sampling_weights)
        dataset = self.datasets[dataset_index]
        step_pos = self._step_pos[dataset_index]
        # re-update
        if step_pos >= len(dataset):
            order = np.arange(len(dataset))
            seed = safe_hash((self.epoch, dataset_index, self.seed, step_pos))
            rng = np.random.default_rng(seed)
            rng.shuffle(order)
            self._step_order[dataset_index] = order
            step_pos = 0

        single_step_index = self._step_order[dataset_index][step_pos]
        # print(f"Single step:{single_step_index}")
        self._step_pos[dataset_index] = step_pos + 1
        return dataset[int(single_step_index)]
    
    def prepare_action_state(self, item):
        if "game" in item["dataset_name"]:
            item["action"] = F.pad(
                    item["action"],
                    (44 + 6, 0),   # 对最后一维：左 pad 44个0，右 pad 0个0
                    mode="constant",
                    value=0
                )
            item["observation.state"] = F.pad(
                    item["observation.state"],
                    (46 + 6, 0),   # 对最后一维：左 pad 46个0，右 pad 0个0
                    mode="constant",
                    value=0
                )
        if "rh20t" in item["dataset_name"]:
            chunk_len = item["action"].shape[0]
            new_action = torch.ones((chunk_len, self.max_action_dim))
            new_action[:, :6] = item["action"][:, :6]
            new_action[:, 6:6 + 1] = item["action"][:, -2:-1]
            # force data
            new_action[:, 44:44 + 6] = item["action"][:, 6:6 + 6]
            new_state = torch.ones(self.max_state_dim)
            new_state[:7] = item["observation.state"][:7]
            new_state[7:7 + 1] = item["observation.state"][-2:-1]
            # force data
            new_state[46:46 + 6] = item["observation.state"][7:7 + 6]
            item["action"] = new_action
            item["observation.state"] = new_state
        
        item["action"] = self.pad_vector(item["action"], self.max_action_dim)
        item["observation.state"] = self.pad_vector(item["observation.state"], self.max_state_dim)
        return item
    
    def norm_data_with_quantile(self, item):
        key1 = "q01"
        key2 = "q99"
        state_q01 = torch.ones(self.max_state_dim) * -1
        state_q99 = torch.ones(self.max_state_dim)
        action_q01 = torch.ones(self.max_action_dim) * -1
        action_q99 = torch.ones(self.max_action_dim)
        action_mask = torch.zeros(self.max_action_dim)
        action_start_dim = 0
        action_end_dim = 0
        state_start_dim = 0
        state_end_dim = 0
        if "agi" in item['dataset_name']:
            action_end_dim = 14
            state_end_dim = 16
        elif "ego_dex" in item['dataset_name']:
            action_start_dim = 0
            action_end_dim = 14 + 30
            state_start_dim = 0
            state_end_dim = 16 + 30
        elif "game" in item["dataset_name"]:
            action_start_dim = 14 + 30
            action_end_dim = 14 + 30 + 50
            state_start_dim = 16 + 30
            state_end_dim = 16 + 30 + 50
        else:
            action_end_dim = 7
            state_end_dim = 8
        
        state_q01[state_start_dim:state_end_dim] = self.stats["observation.state"][key1][state_start_dim:state_end_dim]
        state_q99[state_start_dim:state_end_dim] = self.stats["observation.state"][key2][state_start_dim:state_end_dim]
        action_q01[action_start_dim:action_end_dim] = self.stats["action"][key1][action_start_dim:action_end_dim]
        action_q99[action_start_dim:action_end_dim] = self.stats["action"][key2][action_start_dim:action_end_dim]
        # action
        denom = action_q99 - action_q01
        denom = torch.where(
            denom == 0, torch.tensor(1e-8), denom
        )
        item["action"] = 2.0 * (item["action"] - action_q01) / denom - 1.0
        
        # state
        denom = state_q99 - state_q01
        denom = torch.where(
            denom == 0, torch.tensor(1e-8), denom
        )
        item["observation.state"] = 2.0 * (item["observation.state"] - state_q01) / denom - 1.0
        return item
    
    def __getitem__(self, index):
        # every item key contains t-t+chunk_size elements (large than episode length use repeat last)
        if self.stage == "pretrain":
            item = self.sample_step(index)
        else:
            item = self.full_dataset[index]
        
        # prepare state and action
        item = self.prepare_action_state(item)
        item = self.norm_data_with_quantile(item) # follow cosmos policy
        
        # unified the image keys
        dataset_name = item["dataset_name"]
        data_config = OXE_DATASET_CONFIGS[dataset_name]
        image_obs_keys = data_config["image_obs_keys"] # contain new_key: old_key mapping, such as "primary": "image", ...
        key_to_pad = []
        for new_key, old_key in image_obs_keys.items():
            if old_key != None:
                item[f"observation.images.{new_key}"] = copy.deepcopy(item[f"observation.images.{old_key}"])
                exist_image = item[f"observation.images.{old_key}"]
                if new_key != old_key:
                    del item[f"observation.images.{old_key}"]
            else:
                # if missing, use zero image
                key_to_pad.append(new_key)
        
        for new_key in key_to_pad:
            item[f"observation.images.{new_key}"] = np.zeros_like(exist_image)
        
        # Prepare data for cosmos policy
        
        # Initialize list to store all images
        image_list = []
        current_sequence_idx = 0  # Used to track which sequence of images we are on
        # Get blank array for the first input frame (needed for the tokenizer)
        # Do not duplicate this image
        IMAGE_PRIMARY = "observation.images.primary"
        IMAGE_SECOND = "observation.images.secondary"
        IMAGE_WRIST = "observation.images.wrist"
        CURRENT_IDX = 0
        FUTURE_IDX = -1
        first_input_image = np.expand_dims(np.zeros_like(item[IMAGE_PRIMARY][CURRENT_IDX]), axis=0)
        image_list.append(first_input_image)
        current_sequence_idx += 1
        
        # current state
        if self.use_proprio:
            proprio = item[OBS_ROBOT][CURRENT_IDX]
            # Proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(item[IMAGE_PRIMARY][CURRENT_IDX])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            current_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        
        if self.use_wrist_images:
            wrist_image = item[IMAGE_WRIST][CURRENT_IDX]
            # Duplicate wrist image
            wrist_image = duplicate_array(wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(wrist_image)
            current_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add current third-person image
        if self.use_third_person_images:
            current_primary_image = item[IMAGE_PRIMARY][CURRENT_IDX]
            current_primary_image = duplicate_array(current_primary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_primary_image)
            current_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            
            current_secondary_image = item[IMAGE_SECOND][CURRENT_IDX]
            current_secondary_image = duplicate_array(current_secondary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(current_secondary_image)
            current_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            
        # Add blank image for action chunk
        blank_image = np.zeros_like(item[IMAGE_PRIMARY][CURRENT_IDX])
        # Duplicate blank image
        blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
        image_list.append(blank_image)
        action_latent_idx = current_sequence_idx
        current_sequence_idx += 1
        
        # future state
        
        # Add future proprio
        if self.use_proprio:
            future_proprio = item[OBS_ROBOT][FUTURE_IDX]
            # Not using proprio image; proprio values will be injected into latent diffusion sequence later
            # For now just add blank image
            blank_image = np.zeros_like(item[IMAGE_PRIMARY][FUTURE_IDX])
            blank_image = duplicate_array(blank_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(blank_image)
            future_proprio_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future wrist image
        if self.use_wrist_images:
            future_wrist_image = item[IMAGE_WRIST][FUTURE_IDX]
            future_wrist_image = duplicate_array(future_wrist_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_wrist_image)
            future_wrist_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1

        # Add future third-person image
        if self.use_third_person_images:
            future_primary_image = item[IMAGE_PRIMARY][FUTURE_IDX]
            future_primary_image = duplicate_array(future_primary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_primary_image)
            future_image_latent_idx = current_sequence_idx
            current_sequence_idx += 1
            
            future_secondary_image = item[IMAGE_SECOND][FUTURE_IDX]
            future_secondary_image = duplicate_array(future_secondary_image, total_num_copies=self.num_duplicates_per_image)
            image_list.append(future_secondary_image)
            future_image2_latent_idx = current_sequence_idx
            current_sequence_idx += 1
        
        # Stack images and preprocess
        images = np.concatenate(image_list, axis=0)
        # print(len(image_list), images.shape)
        images = preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
        )
        # print(images.shape) # torch.Size([37, 3, 256, 256])
        action_chunk = item["action"] # pad with last action
        # print(proprio.shape, future_proprio.shape) # 128 128
        
        sample_dict = {
            "video": images,
            "actions": action_chunk,
            "task": item["task"],
            "t5_text_mask": torch.ones(512, dtype=torch.int64),  # Just copying what others have done in this codebase
            "fps": 16,  # Just set to some fixed value since we aren't generating videos anyway
            "padding_mask": torch.zeros(
                1, self.final_image_size, self.final_image_size
            ),  # Just copying what others have done in this codebase
            "image_size": self.final_image_size
            * torch.ones(
                4
            ),  # Just copying what others have done in this codebase; important because it shows up as model input
            "proprio": proprio if self.use_proprio else torch.zeros_like(item[OBS_ROBOT][CURRENT_IDX]),
            "future_proprio": future_proprio if self.use_proprio else torch.zeros_like(item[OBS_ROBOT][FUTURE_IDX]),
            "__key__": index,  # Unique sample identifier (required for callbacks)
            
            # "rollout_data_mask": rollout_data_mask,
            # "rollout_data_success_mask": rollout_data_success_mask,
            # "world_model_sample_mask": 1 if is_world_model_sample else 0,
            # "value_function_sample_mask": 1 if is_value_function_sample else 0,
            # "global_rollout_idx": global_rollout_idx,
            "action_latent_idx": action_latent_idx,
            # "value_latent_idx": value_latent_idx if self.return_value_function_returns else -1,
            "current_proprio_latent_idx": current_proprio_latent_idx if self.use_proprio else -1,
            "current_wrist_image_latent_idx": current_wrist_image_latent_idx if self.use_wrist_images else -1,
            "current_image_latent_idx": current_image_latent_idx if self.use_third_person_images else -1,
            "current_image2_latent_idx": current_image2_latent_idx if self.use_third_person_images else -1,
            "future_proprio_latent_idx": future_proprio_latent_idx if self.use_proprio else -1,
            "future_wrist_image_latent_idx": future_wrist_image_latent_idx if self.use_wrist_images else -1,
            "future_image_latent_idx": future_image_latent_idx if self.use_third_person_images else -1,
            "future_image2_latent_idx": future_image2_latent_idx if self.use_third_person_images else -1,
            # "value_function_return": value_function_return,
            # "next_action_chunk": next_action_chunk,
            # "next_value_function_return": next_value_function_return,
        }
        
        return sample_dict


def resolve_delta_timestamps(
    cfg: PreTrainedConfig, ds_meta: LeRobotDatasetMetadata
) -> dict[str, list] | None:
    """Resolves delta_timestamps by reading from the 'delta_indices' properties of the PreTrainedConfig.

    Args:
        cfg (PreTrainedConfig): The PreTrainedConfig to read delta_indices from.
        ds_meta (LeRobotDatasetMetadata): The dataset from which features and fps are used to build
            delta_timestamps against.

    Returns:
        dict[str, list] | None: A dictionary of delta_timestamps, e.g.:
            {
                "observation.state": [-0.04, -0.02, 0]
                "observation.action": [-0.02, 0, 0.02]
            }
            returns `None` if the the resulting dict is empty.
    """
    delta_timestamps = {}
    for key in ds_meta.features:
        if key == "next.reward" and cfg.reward_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.reward_delta_indices]
        if key == "action" and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps

@parser.wrap()
def dataset_func_test(cfg: TrainPipelineConfig):
    cfg.validate()
    cfg.dataset.parent_dir="/data_16T/lerobot_openx/"
    cfg.dataset.processor="/datassd_1T/qwen25vl/Qwen2.5-VL-7B-Instruct/"
    
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms)
    )
    
    dataset = MultiDatasetforDistTraining(
        cfg=cfg,
        image_transforms=image_transforms,
        seed=cfg.seed,
        data_mix="oxe_magic_soup_plus",
        vla2root_json="vla2root_bak_single.json"
    )
    
    item = dataset[0]
    for key, value in item.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}")
        else:
            print(f"{key}: {value}")
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=extra_collate_fn,
        batch_size=2
    )
    dl_iter = cycle(dataloader)
    batch = next(dl_iter)
    keys = list(batch.keys())
    print(f"batch:{keys}")
    for key in keys:
        print(f"Value type for key {key}: {type(batch[key])}")
        if isinstance(batch[key], torch.Tensor):
            print(f"Tensor shape: {batch[key].shape}")
        
        if isinstance(batch[key], list):
            print(f"List elements: {type(batch[key][0])}")
            print(f"List actual value: {batch[key]}")
    # print(f"Video shape: {batch['observation.images.secondary'].shape}")

    # print(dataset)
    # for i in range(1):
    #     item = dataset[i]
    #     print(f"item {i}:")
    #     for key, value in item.items():
    #         print(f"{key}")
    #         if key[:18] == "observation.images":
    #             print(f"{key}: {value.shape}")
    #             if value.ndim == 4:
    #                 for img in value:
    #                     print(img.shape)
    #     print("\n")
        
def extra_collate_fn(batch):
    collated = {}
    key_to_pad = ["input_ids", "attention_mask", "labels"]
    key_to_default_collate = ["observation.state", "action"]
    key_to_append_to_list = ["second_per_grid_ts"]
    for key in batch[0].keys():
        items = [sample[key] for sample in batch]
            
        if key in key_to_pad:
            max_length = max([item.shape[1] for item in items])
            padded_tensor = []
            for item in items:
                if item.shape[1] < max_length:
                    pad_size = max_length - item.shape[1]
                    padded_tensor.append(torch.nn.functional.pad(item, (pad_size, 0), value=PAD_VALUE[key]))
                else:
                    padded_tensor.append(item)
            item = torch.cat(padded_tensor, dim=0)
            collated[key] = item
        elif isinstance(items[0],torch.Tensor) and key not in key_to_default_collate:
            item = torch.cat(items, dim=0)
            collated[key] = item
        elif key in key_to_append_to_list:
            collated_item = []
            for item in items:
                collated_item.append(item[0])
            collated[key] = collated_item
        else:
            collated[key] = default_collate(items)
    
    return collated
        
    
if __name__ == "__main__":
    dataset_func_test()