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
import concurrent.futures
import contextlib
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import PIL.Image
import pyarrow.parquet as pq
import torch
import torch.utils
from torch.utils.data import ConcatDataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RevisionNotFoundError

from lerobot.common.datasets_v30.compute_stats import compute_episode_stats, aggregate_stats_with_padding
from lerobot.common.datasets_v30.dataset_metadata import CODEBASE_VERSION, LeRobotDatasetMetadata
from lerobot.common.datasets_v30.feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
    validate_episode_buffer,
    validate_frame,
)
from lerobot.common.datasets_v30.image_writer import AsyncImageWriter, write_image
from lerobot.common.datasets_v30.io_utils import (
    embed_images,
    get_file_size_in_mb,
    hf_transform_to_torch,
    load_episodes,
    load_nested_dataset,
    write_info,
)
from lerobot.common.datasets_v30.utils import (
    DEFAULT_EPISODES_PATH,
    DEFAULT_IMAGE_PATH,
    create_lerobot_dataset_card,
    get_safe_version,
    is_valid_version,
    update_chunk_file_indices,
)
from lerobot.common.datasets_v30.video_utils import (
    StreamingVideoEncoder,
    concatenate_video_files,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_duration_in_s,
    resolve_vcodec,
)
from lerobot.common.datasets_v30.constants import HF_LEROBOT_HOME
from lerobot.configs.policies import PreTrainedConfig
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
from lerobot.common.datasets.oxe_configs import OXE_DATASET_CONFIGS

import random
from PIL import Image
import json
import os
from tabulate import tabulate
from datetime import datetime
import copy
import math

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"Random seed set to: {seed}")

def _encode_video_worker(
    video_key: str,
    episode_index: int,
    root: Path,
    fps: int,
    vcodec: str = "libsvtav1",
    encoder_threads: int | None = None,
) -> Path:
    temp_path = Path(tempfile.mkdtemp(dir=root)) / f"{video_key}_{episode_index:03d}.mp4"
    fpath = DEFAULT_IMAGE_PATH.format(image_key=video_key, episode_index=episode_index, frame_index=0)
    img_dir = (root / fpath).parent
    encode_video_frames(
        img_dir, temp_path, fps, vcodec=vcodec, overwrite=True, encoder_threads=encoder_threads
    )
    shutil.rmtree(img_dir)
    return temp_path


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[str, list[float]] | None = None,
        tolerance_s: float = 2e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
        dataset_name: str = None,
        video_return_type: str = "float32",
        decode_device: str | None = None,
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
              codebase_version v3.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v2.1 to v3.0, which you can find at
              lerobot/scripts/convert_dataset_v21_to_v30.py.


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
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── file-000.parquet
        │   │   ├── file-001.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes
        │   │   ├── chunk-000
        │   │   │   ├── file-000.parquet
        │   │   │   ├── file-001.parquet
        │   │   │   └── ...
        │   │   ├── chunk-001
        │   │   │   └── ...
        │   │   └── ...
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.parquet
        └── videos
            ├── observation.images.laptop
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            ├── observation.images.phone
            │   ├── chunk-000
            │   │   ├── file-000.mp4
            │   │   ├── file-001.mp4
            │   │   └── ...
            │   ├── chunk-001
            │   │   └── ...
            │   └── ...
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. Multiple episodes are
        consolidated into chunked files which improves storage efficiency and loading performance. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset.
            root (Path | None, optional): Local directory where the dataset will be downloaded and
                stored. If set, all dataset files will be stored directly under this path. If not set, the
                dataset files will be stored under $HF_LEROBOT_HOME/repo_id (configurable via the
                HF_LEROBOT_HOME environment variable).
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
            force_cache_sync (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
            batch_encoding_size (int, optional): Number of episodes to accumulate before batch encoding videos.
                Set to 1 for immediate encoding (default), or higher for batched encoding. Defaults to 1.
            vcodec (str, optional): Video codec for encoding videos during recording. Options: 'h264', 'hevc',
                'libsvtav1', 'auto', or hardware-specific codecs like 'h264_videotoolbox', 'h264_nvenc'.
                Defaults to 'libsvtav1'. Use 'auto' to auto-detect the best available hardware encoder.
            streaming_encoding (bool, optional): If True, encode video frames in real-time during capture
                instead of writing PNG images first. This makes save_episode() near-instant. Defaults to False.
            encoder_queue_maxsize (int, optional): Maximum number of frames to buffer per camera when using
                streaming encoding. Defaults to 30 (~1s at 30fps).
            encoder_threads (int | None, optional): Number of threads per encoder instance. None lets the
                codec auto-detect (default). Lower values reduce CPU usage per encoder. Maps to 'lp' (via svtav1-params) for
                libsvtav1 and 'threads' for h264/hevc.
        """
        super().__init__()
        self.repo_id = repo_id
        self.dataset_name = dataset_name
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.video_return_type = video_return_type
        self.decode_device = decode_device or "cpu"
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self._video_keys_to_decode = None
        self.episodes_since_last_encoding = 0
        self.vcodec = resolve_vcodec(vcodec)
        self._encoder_threads = encoder_threads

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None
        self.writer = None
        self.latest_episode = None
        self._current_file_start_frame = None  # Track the starting frame index of the current parquet file
        self._streaming_encoder = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync
        )

        # Track dataset state for efficient incremental writing
        self._lazy_loading = False
        self._recorded_frames = self.meta.total_frames
        self._writer_closed_for_reading = False

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.hf_dataset = self.load_hf_dataset()
            # Check if cached dataset contains all requested episodes
            if not self._check_cached_episodes_sufficient():
                raise FileNotFoundError("Cached dataset doesn't contain all requested episodes")
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            self.download(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        # Create mapping from absolute indices to relative indices when only a subset of the episodes are loaded
        # Build a mapping: absolute_index -> relative_index_in_filtered_dataset
        self._absolute_to_relative_idx = None
        if self.episodes is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        # Initialize streaming encoder for resumed recording
        if streaming_encoding and len(self.meta.video_keys) > 0:
            self._streaming_encoder = StreamingVideoEncoder(
                fps=self.meta.fps,
                vcodec=self.vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                queue_maxsize=encoder_queue_maxsize,
                encoder_threads=encoder_threads,
            )

    def _close_writer(self) -> None:
        """Close and cleanup the parquet writer if it exists."""
        writer = getattr(self, "writer", None)
        if writer is not None:
            writer.close()
            self.writer = None

    def __del__(self):
        """
        Trust the user to call .finalize() but as an added safety check call the parquet writer to stop when calling the destructor
        """
        self._close_writer()

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

        card = create_lerobot_dataset_card(
            tags=tags, dataset_info=self.meta.info, license=license, repo_id=self.repo_id, **card_kwargs
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

    def download(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        ignore_patterns = None if download_videos else "videos/"
        files = None
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
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        # Get available episode indices from cached dataset
        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        # Determine requested episodes
        if self.episodes is None:
            requested_episodes = set(range(self.meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        # Check if all requested episodes are available in cached data
        if not requested_episodes.issubset(available_episodes):
            return False

        # Check if all required video files exist
        if len(self.meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self.meta.video_keys:
                    video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes.

        Note: When episodes a subset of the full dataset is requested, we must return the
        actual loaded data length (len(self.hf_dataset)) rather than metadata total_frames.
        self.meta.total_frames is the total number of frames in the full dataset.
        """
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self.meta.total_frames

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

    @property
    def video_keys_to_decode(self) -> list[str]:
        """Video keys that ``__getitem__`` actually decodes.

        Defaults to every video key. Set ``dataset.video_keys_to_decode = [...]`` to skip
        decoding cameras the training pipeline does not consume, which is a large speed-up
        on datasets that ship many streams (e.g. 6 tactile cameras + 3 RGB cameras).
        """
        if self._video_keys_to_decode is None:
            return self.meta.video_keys
        return self._video_keys_to_decode

    @video_keys_to_decode.setter
    def video_keys_to_decode(self, keys: list[str] | None) -> None:
        if keys is None:
            self._video_keys_to_decode = None
            return
        available = set(self.meta.video_keys)
        self._video_keys_to_decode = [k for k in keys if k in available]

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps.

        Args:
            abs_idx: The absolute index in the full dataset (not the relative index in filtered episodes).
            ep_idx: The episode index.

        Returns:
            A tuple of (query_indices, padding) where:
            - query_indices: Dict mapping keys to lists of absolute indices to query
            - padding: Dict mapping "{key}_is_pad" to boolean tensors indicating padded positions
        """
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
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
        for key in self.video_keys_to_decode:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                else:
                    relative_indices = list(query_indices[key])
                timestamps = self.hf_dataset[relative_indices]["timestamp"]
                query_timestamps[key] = torch.stack(list(timestamps)).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """
        Query dataset for indices across keys, skipping video keys.

        Uses direct list indexing, which goes through arrow's fast gather path.
        This is dramatically faster than `dataset.select(indices)[key]`, which
        materializes an on-disk indices mapping for non-contiguous index lists
        (~2.6s per call on a 31M-row concatenated dataset).

        Args:
            query_indices: Dict mapping keys to index lists to retrieve

        Returns:
            Dict with stacked tensors of queried data (video keys excluded)
        """
        result: dict = {}
        if not query_indices:
            return result
        # Group all non-video keys and fetch rows once per unique index list.
        for key, q_idx in query_indices.items():
            if key in self.meta.video_keys:
                continue
            # Map absolute indices to relative indices if needed
            relative_indices = (
                list(q_idx)
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            # NOTE: `hf_dataset.select(...)` is extremely slow for non-contiguous
            # index lists on large concatenated datasets (it materializes an arrow
            # indices mapping, ~2.6s for a 31M-row dataset). Plain list indexing
            # goes through the fast `fast_gather` path instead (~0.4ms).
            try:
                rows = self.hf_dataset[relative_indices][key]
                result[key] = rows if isinstance(rows, torch.Tensor) else torch.stack(list(rows))
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack([self.hf_dataset[i][key] for i in relative_indices])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        ep = self.meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            # Episodes are stored sequentially on a single mp4 to reduce the number of files.
            # Thus we load the start timestamp of the episode on this mp4 and,
            # shift the query timestamp accordingly.
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(
                video_path,
                shifted_query_ts,
                self.tolerance_s,
                self.video_backend,
                return_type=self.video_return_type,
                device=self.decode_device,
            )
            item[vid_key] = frames.squeeze(0)
        return item

    def _ensure_hf_dataset_loaded(self):
        """Lazy load the HF dataset only when needed for reading."""
        if self._lazy_loading or self.hf_dataset is None:
            # Close the writer before loading to ensure parquet file is properly finalized
            if self.writer is not None:
                self._close_writer()
                self._writer_closed_for_reading = True
            self.hf_dataset = self.load_hf_dataset()
            self._lazy_loading = False

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        # import time
        # t0 = time.perf_counter()
        # Ensure dataset is loaded when we actually need to read from it
        self._ensure_hf_dataset_loaded()
        # t1 = time.perf_counter()
        item = self.hf_dataset[idx]
        # t2 = time.perf_counter()
        ep_idx = item["episode_index"].item()
        # Use the absolute index from the dataset for delta timestamp calculations
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val
        # t3 = time.perf_counter()
        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}
        # t4 = time.perf_counter()

        if self.image_transforms is not None:
            image_keys = [key for key in self.meta.camera_keys if key in item]
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])
        # t5 = time.perf_counter()

        # Add task as a string
        task_idx = item["task_index"].item()
        # print(len(self.meta.tasks), task_idx)
        item["task"] = self.meta.tasks.iloc[task_idx].name
        # add subtask information if available
        if "subtask_index" in self.features and self.meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self.meta.subtasks.iloc[subtask_idx].name

        item["fps"] = math.ceil(self.meta.fps)
        item["dataset_name"] = self.dataset_name
        # t6 = time.perf_counter()

        if "action" not in item.keys():
            candidate_state_keys = ["observation.ee_ort6d_pos", "observations.ee_ort6d_pos", "observations.ee_6d_pos"]
            for key in candidate_state_keys:
                if key in item.keys():
                    item["observation.state"] = item[key]
                    break
            candidate_action_keys = ["action.ee_ort6d_pos", "action.ee_6d_pos"]
            for key in candidate_action_keys:
                if key in item.keys():
                    item["action"] = item[key]
                    break
            # item["observation.state"] = item["observation.ee_ort6d_pos"]
        # t7 = time.perf_counter()

        keys = list(item.keys())
        for key in keys:
            if item[key] is None:
                del item[key]  # Remove keys with None values (e.g. video keys when videos are not downloaded)
                # item[key] = torch.tensor([])  # Replace None with empty tensor for consistency
        # t8 = time.perf_counter()
        # print(
        #     f"[LeRobotDataset.__getitem__] "
        #     f"_ensure_hf_dataset_loaded: {(t1 - t0) * 1000:.3f}ms, "
        #     f"hf_dataset[idx]: {(t2 - t1) * 1000:.3f}ms, "
        #     f"delta_indices+query+video: {(t4 - t2) * 1000:.3f}ms, "
        #     f"image_transforms: {(t5 - t4) * 1000:.3f}ms, "
        #     f"task/metadata: {(t7 - t5) * 1000:.3f}ms, "
        #     f"cleanup None keys: {(t8 - t7) * 1000:.3f}ms, "
        #     f"total: {(t8 - t0) * 1000:.3f}ms"
        # )
        return item
        
        # return self._finalize_item(item)

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

    def finalize(self):
        """
        Close the parquet writers. This function needs to be called after data collection/conversion, else footer metadata won't be written to the parquet files.
        The dataset won't be valid and can't be loaded as ds = LeRobotDataset(repo_id=repo, root=HF_LEROBOT_HOME.joinpath(repo))
        """
        self._close_writer()
        self.meta._close_writer()
        if self._streaming_encoder is not None:
            self._streaming_encoder.close()

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    # TODO(Steven): consider move this to utils
    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _get_image_file_dir(self, episode_index: int, image_key: str) -> Path:
        return self._get_image_file_path(episode_index, image_key, frame_index=0).parent

    def _save_image(
        self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path, compress_level: int = 1
    ) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath, compress_level=compress_level)
        else:
            self.image_writer.save_image(image=image, fpath=fpath, compress_level=compress_level)

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
        self.episode_buffer["task"].append(frame.pop("task"))  # Remove task from frame after processing

        # Start streaming encoder on first frame of episode (once, before iterating keys)
        if frame_index == 0 and self._streaming_encoder is not None:
            self._streaming_encoder.start_episode(
                video_keys=list(self.meta.video_keys),
                temp_dir=self.root,
            )

        # Add frame features to episode_buffer
        for key in frame:
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] == "video" and self._streaming_encoder is not None:
                self._streaming_encoder.feed_frame(key, frame[key])
                self.episode_buffer[key].append(None)  # Placeholder (video keys are skipped in parquet)
            elif self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                compress_level = 1 if self.features[key]["dtype"] == "video" else 6
                self._save_image(frame[key], img_path, compress_level)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(
        self,
        episode_data: dict | None = None,
        parallel_encoding: bool = True,
    ) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Video encoding is handled automatically based on batch_encoding_size:
        - If batch_encoding_size == 1: Videos are encoded immediately after each episode
        - If batch_encoding_size > 1: Videos are encoded in batches.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
            parallel_encoding (bool, optional): If True, encode videos in parallel using ProcessPoolExecutor.
                Defaults to True on Linux, False on macOS as it tends to use all the CPU available already.
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Update tasks and task indices with new tasks if any
        self.meta.save_episode_tasks(episode_tasks)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        # Wait for image writer to end, so that episode stats over images can be computed
        self._wait_image_writer()

        has_video_keys = len(self.meta.video_keys) > 0
        use_streaming = self._streaming_encoder is not None and has_video_keys
        use_batched_encoding = self.batch_encoding_size > 1

        if use_streaming:
            # Compute stats for non-video features only (video stats come from encoder)
            non_video_buffer = {
                k: v
                for k, v in episode_buffer.items()
                if self.features.get(k, {}).get("dtype") not in ("video",)
            }
            non_video_features = {k: v for k, v in self.features.items() if v["dtype"] != "video"}
            ep_stats = compute_episode_stats(non_video_buffer, non_video_features)
        else:
            ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)

        if use_streaming:
            # Finish streaming encoding and collect results
            streaming_results = self._streaming_encoder.finish_episode()
            for video_key in self.meta.video_keys:
                temp_path, video_stats = streaming_results[video_key]
                if video_stats is not None:
                    # Format stats same as compute_episode_stats: normalize to [0,1], reshape to (C,1,1)
                    ep_stats[video_key] = {
                        k: v if k == "count" else np.squeeze(v.reshape(1, -1, 1, 1) / 255.0, axis=0)
                        for k, v in video_stats.items()
                    }
                ep_metadata.update(self._save_episode_video(video_key, episode_index, temp_path=temp_path))
        elif has_video_keys and not use_batched_encoding:
            num_cameras = len(self.meta.video_keys)
            if parallel_encoding and num_cameras > 1:
                # TODO(Steven): Ideally we would like to control the number of threads per encoding such that:
                # num_cameras * num_threads = (total_cpu -1)
                with concurrent.futures.ProcessPoolExecutor(max_workers=num_cameras) as executor:
                    future_to_key = {
                        executor.submit(
                            _encode_video_worker,
                            video_key,
                            episode_index,
                            self.root,
                            self.fps,
                            self.vcodec,
                            self._encoder_threads,
                        ): video_key
                        for video_key in self.meta.video_keys
                    }

                    results = {}
                    for future in concurrent.futures.as_completed(future_to_key):
                        video_key = future_to_key[future]
                        try:
                            temp_path = future.result()
                            results[video_key] = temp_path
                        except Exception as exc:
                            logger.error(f"Video encoding failed for {video_key}: {exc}")
                            raise exc

                for video_key in self.meta.video_keys:
                    temp_path = results[video_key]
                    ep_metadata.update(
                        self._save_episode_video(video_key, episode_index, temp_path=temp_path)
                    )
            else:
                for video_key in self.meta.video_keys:
                    ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # `meta.save_episode` need to be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # Check if we should trigger batch encoding
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # Reset episode buffer and clean up temporary images (if not already deleted during video encoding)
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _batch_save_episode_video(self, start_episode: int, end_episode: int | None = None) -> None:
        """
        Batch save videos for multiple episodes.

        Args:
            start_episode: Starting episode index (inclusive)
            end_episode: Ending episode index (exclusive). If None, encodes all episodes from start_episode to the current episode.
        """
        if end_episode is None:
            end_episode = self.num_episodes

        logger.info(
            f"Batch encoding {self.batch_encoding_size} videos for episodes {start_episode} to {end_episode - 1}"
        )

        chunk_idx = self.meta.episodes[start_episode]["data/chunk_index"]
        file_idx = self.meta.episodes[start_episode]["data/file_index"]
        episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        episode_df = pd.read_parquet(episode_df_path)

        for ep_idx in range(start_episode, end_episode):
            logger.info(f"Encoding videos for episode {ep_idx}")

            if (
                self.meta.episodes[ep_idx]["data/chunk_index"] != chunk_idx
                or self.meta.episodes[ep_idx]["data/file_index"] != file_idx
            ):
                # The current episode is in a new chunk or file.
                # Save previous episode dataframe and update the Hugging Face dataset by reloading it.
                episode_df.to_parquet(episode_df_path)
                self.meta.episodes = load_episodes(self.root)

                # Load new episode dataframe
                chunk_idx = self.meta.episodes[ep_idx]["data/chunk_index"]
                file_idx = self.meta.episodes[ep_idx]["data/file_index"]
                episode_df_path = self.root / DEFAULT_EPISODES_PATH.format(
                    chunk_index=chunk_idx, file_index=file_idx
                )
                episode_df = pd.read_parquet(episode_df_path)

            # Save the current episode's video metadata to the dataframe
            video_ep_metadata = {}
            for video_key in self.meta.video_keys:
                video_ep_metadata.update(self._save_episode_video(video_key, ep_idx))
            video_ep_metadata.pop("episode_index")
            video_ep_df = pd.DataFrame(video_ep_metadata, index=[ep_idx]).convert_dtypes(
                dtype_backend="pyarrow"
            )  # allows NaN values along with integers

            episode_df = episode_df.combine_first(video_ep_df)
            episode_df.to_parquet(episode_df_path)
            self.meta.episodes = load_episodes(self.root)

    def _save_episode_data(self, episode_buffer: dict) -> dict:
        """Save episode data to a parquet file and update the Hugging Face dataset of frames data.

        This function processes episodes data from a buffer, converts it into a Hugging Face dataset,
        and saves it as a parquet file. It handles both the creation of new parquet files and the
        updating of existing ones based on size constraints. After saving the data, it reloads
        the Hugging Face dataset to ensure it is up-to-date.

        Notes: We both need to update parquet files and HF dataset:
        - `pandas` loads parquet file in RAM
        - `datasets` relies on a memory mapping from pyarrow (no RAM). It either converts parquet files to a pyarrow cache on disk,
          or loads directly from pyarrow cache.
        """
        # Convert buffer into HF Dataset
        ep_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(ep_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        ep_num_frames = len(ep_dataset)

        if self.latest_episode is None:
            # Initialize indices and frame count for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            global_frame_index = 0
            self._current_file_start_frame = 0
            # However, if the episodes already exists
            # It means we are resuming recording, so we need to load the latest episode
            # Update the indices to avoid overwriting the latest episode
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                latest_ep = self.meta.episodes[-1]
                global_frame_index = latest_ep["dataset_to_index"]
                chunk_idx = latest_ep["data/chunk_index"]
                file_idx = latest_ep["data/file_index"]

                # When resuming, move to the next file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                self._current_file_start_frame = global_frame_index
        else:
            # Retrieve information from the latest parquet file
            latest_ep = self.latest_episode
            chunk_idx = latest_ep["data/chunk_index"]
            file_idx = latest_ep["data/file_index"]
            global_frame_index = latest_ep["index"][-1] + 1

            latest_path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
            latest_size_in_mb = get_file_size_in_mb(latest_path)

            frames_in_current_file = global_frame_index - self._current_file_start_frame
            av_size_per_frame = (
                latest_size_in_mb / frames_in_current_file if frames_in_current_file > 0 else 0
            )

            # Determine if a new parquet file is needed
            if (
                latest_size_in_mb + av_size_per_frame * ep_num_frames >= self.meta.data_files_size_in_mb
                or self._writer_closed_for_reading
            ):
                # Size limit is reached or writer was closed for reading, prepare new parquet file
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                self._close_writer()
                self._writer_closed_for_reading = False
                self._current_file_start_frame = global_frame_index

        ep_dict["data/chunk_index"] = chunk_idx
        ep_dict["data/file_index"] = file_idx

        # Write the resulting dataframe from RAM to disk
        path = self.root / self.meta.data_path.format(chunk_index=chunk_idx, file_index=file_idx)
        path.parent.mkdir(parents=True, exist_ok=True)

        table = ep_dataset.with_format("arrow")[:]
        if not self.writer:
            self.writer = pq.ParquetWriter(
                path, schema=table.schema, compression="snappy", use_dictionary=True
            )
        self.writer.write_table(table)

        metadata = {
            "data/chunk_index": chunk_idx,
            "data/file_index": file_idx,
            "dataset_from_index": global_frame_index,
            "dataset_to_index": global_frame_index + ep_num_frames,
        }

        # Store metadata with episode data for next episode
        self.latest_episode = {**ep_dict, **metadata}

        # Mark that the HF dataset needs reloading (lazy loading approach)
        # This avoids expensive reloading during sequential recording
        self._lazy_loading = True
        # Update recorded frames count for efficient length tracking
        self._recorded_frames += ep_num_frames

        return metadata

    def _save_episode_video(
        self,
        video_key: str,
        episode_index: int,
        temp_path: Path | None = None,
    ) -> dict:
        # Encode episode frames into a temporary video
        if temp_path is None:
            ep_path = self._encode_temporary_episode_video(video_key, episode_index)
        else:
            ep_path = temp_path

        ep_size_in_mb = get_file_size_in_mb(ep_path)
        ep_duration_in_s = get_video_duration_in_s(ep_path)

        if (
            episode_index == 0
            or self.meta.latest_episode is None
            or f"videos/{video_key}/chunk_index" not in self.meta.latest_episode
        ):
            # Initialize indices for a new dataset made of the first episode data
            chunk_idx, file_idx = 0, 0
            if self.meta.episodes is not None and len(self.meta.episodes) > 0:
                # It means we are resuming recording, so we need to load the latest episode
                # Update the indices to avoid overwriting the latest episode
                old_chunk_idx = self.meta.episodes[-1][f"videos/{video_key}/chunk_index"]
                old_file_idx = self.meta.episodes[-1][f"videos/{video_key}/file_index"]
                chunk_idx, file_idx = update_chunk_file_indices(
                    old_chunk_idx, old_file_idx, self.meta.chunks_size
                )
            latest_duration_in_s = 0.0
            new_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(ep_path), str(new_path))
        else:
            # Retrieve information from the latest updated video file using latest_episode
            latest_ep = self.meta.latest_episode
            chunk_idx = latest_ep[f"videos/{video_key}/chunk_index"][0]
            file_idx = latest_ep[f"videos/{video_key}/file_index"][0]

            latest_path = self.root / self.meta.video_path.format(
                video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
            )
            latest_size_in_mb = get_file_size_in_mb(latest_path)
            latest_duration_in_s = latest_ep[f"videos/{video_key}/to_timestamp"][0]

            if latest_size_in_mb + ep_size_in_mb >= self.meta.video_files_size_in_mb:
                # Move temporary episode video to a new video file in the dataset
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, self.meta.chunks_size)
                new_path = self.root / self.meta.video_path.format(
                    video_key=video_key, chunk_index=chunk_idx, file_index=file_idx
                )
                new_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ep_path), str(new_path))
                latest_duration_in_s = 0.0
            else:
                # Update latest video file
                concatenate_video_files(
                    [latest_path, ep_path],
                    latest_path,
                )

        # Remove temporary directory
        shutil.rmtree(str(ep_path.parent))

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if episode_index == 0:
            self.meta.update_video_info(video_key)
            write_info(self.meta.info, self.meta.root)  # ensure video info always written properly

        metadata = {
            "episode_index": episode_index,
            f"videos/{video_key}/chunk_index": chunk_idx,
            f"videos/{video_key}/file_index": file_idx,
            f"videos/{video_key}/from_timestamp": latest_duration_in_s,
            f"videos/{video_key}/to_timestamp": latest_duration_in_s + ep_duration_in_s,
        }
        return metadata

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        # Cancel streaming encoder if active
        if self._streaming_encoder is not None:
            self._streaming_encoder.cancel_episode()

        # Clean up image files for the current episode buffer
        if delete_images:
            # Wait for the async image writer to finish
            if self.image_writer is not None:
                self._wait_image_writer()
            episode_index = self.episode_buffer["episode_index"]
            if isinstance(episode_index, np.ndarray):
                episode_index = episode_index.item() if episode_index.size == 1 else episode_index[0]
            for cam_key in self.meta.image_keys:
                img_dir = self._get_image_file_dir(episode_index, cam_key)
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logger.warning(
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

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        return _encode_video_worker(
            video_key, episode_index, self.root, self.fps, self.vcodec, self._encoder_threads
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        features: dict,
        root: str | Path | None = None,
        robot_type: str | None = None,
        use_videos: bool = True,
        tolerance_s: float = 2e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
        batch_encoding_size: int = 1,
        vcodec: str = "libsvtav1",
        metadata_buffer_size: int = 10,
        streaming_encoding: bool = False,
        encoder_queue_maxsize: int = 30,
        encoder_threads: int | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        vcodec = resolve_vcodec(vcodec)
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            robot_type=robot_type,
            features=features,
            root=root,
            use_videos=use_videos,
            metadata_buffer_size=metadata_buffer_size,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None
        obj.batch_encoding_size = batch_encoding_size
        obj.episodes_since_last_encoding = 0
        obj.vcodec = vcodec
        obj._encoder_threads = encoder_threads

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj._absolute_to_relative_idx = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        obj.writer = None
        obj.latest_episode = None
        obj._current_file_start_frame = None
        # Initialize tracking for incremental recording
        obj._lazy_loading = False
        obj._recorded_frames = 0
        obj._writer_closed_for_reading = False

        # Initialize streaming encoder
        if streaming_encoding and len(obj.meta.video_keys) > 0:
            obj._streaming_encoder = StreamingVideoEncoder(
                fps=fps,
                vcodec=vcodec,
                pix_fmt="yuv420p",
                g=2,
                crf=30,
                preset=None,
                queue_maxsize=encoder_queue_maxsize,
                encoder_threads=encoder_threads,
            )
        else:
            obj._streaming_encoder = None

        return obj



class MultiDatasetforDistTraining(torch.utils.data.Dataset):
    def __init__(self, cfg, image_transforms, seed: int = 1000, 
                 data_mix: str = "toy", vla2root_json: str = None, 
                 banlance_weight=True, is_ft = False,
                 dataset_size_one_epoch = 1000_0000):
        super().__init__()
        self.episodes = None
        self.cfg = cfg
        self.seed = seed
        # set seed
        set_seed(seed)
        # specific process
        # get sample weights
        mixture_spec = OXE_NAMED_MIXTURES[data_mix]
        included_datasets, sample_weights = [], []
        for d_name, d_weight in mixture_spec:
            if d_name in included_datasets:
                print(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
                continue

            included_datasets.append(d_name)
            sample_weights.append(d_weight)
        
        print(included_datasets, sample_weights)
        # get dataset and dataset length
        
        default_parent_dir = "/data_16T/lerobot_openx/"
        parent_dir = self.cfg.dataset.parent_dir
        if self.cfg.dataset.parent_dir is None:
            parent_dir = default_parent_dir
        print(parent_dir)
        # parent_dir = "/mnt/wangxiaofa/robot_dataset/lerobot-format/"
        
        self.datasets = []
        self.dataset_sizes = []
        self.dataset_names = []
        meta_features = None
        with open(vla2root_json, "r") as f:
            vla2data_root = json.load(f)
        for dataset_name in included_datasets:
            if dataset_name in vla2data_root.keys():
                data_root = vla2data_root[dataset_name]
                data_root = os.path.join(parent_dir, data_root)
                print(f"Load data from {data_root}")
                repo_id = f"bulldog-{dataset_name}" # any
                ds_meta = LeRobotDatasetMetadata(repo_id, root=data_root)
                if meta_features == None:
                    meta_features = ds_meta.features
                delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
                dataset = LeRobotDataset(
                    repo_id, 
                    root=data_root,
                    delta_timestamps=delta_timestamps,
                    image_transforms=image_transforms,
                    video_backend=cfg.dataset.video_backend,
                    dataset_name=dataset_name,
                    # video_return_type="float32" # [0,1]
                    video_return_type="uint8" # [0, 255]
                )
                self.datasets.append(dataset)
                self.dataset_sizes.append(len(dataset))
                self.dataset_names.append(dataset_name)
                # del 
            else:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - {dataset_name} not found in vla2root.json, skipping...")
        
        self.is_ft = is_ft
        if is_ft:
            self.id2dataset = {}
            self.num_episodes = 0
            self.dataset_len = 0
            dataset_id = 0
            for i in range(len(self.datasets)):
                dataset = self.datasets[i]
                num_frames = dataset.num_frames
                print(f"Dataset {dataset.dataset_name} has {num_frames} frames.")
                self.dataset_len += num_frames
                num_episodes = dataset.num_episodes
                self.num_episodes += num_episodes
                start_id = self.dataset_len - num_frames
                end_id = self.dataset_len
                data_id = 0
                for index in range(start_id, end_id):
                    self.id2dataset[index] = (dataset_id, data_id)
                    data_id += 1
                dataset_id += 1
                assert data_id == num_frames
            print(f"data mix:{data_mix} has {self.num_episodes} episodes, {self.dataset_len} frames.")
        else:
            if banlance_weight:
                # filter out the datasets not in vla2root.json
                new_sample_weights = [
                    sw 
                    for sw, dataset in zip(sample_weights, included_datasets) 
                    if dataset in vla2data_root.keys()
                ]
                sample_weights = np.array(new_sample_weights) * np.array(self.dataset_sizes)
                print(f"Banlanced:{sample_weights}")
            self.sample_weights = np.array(sample_weights) / np.sum(sample_weights)
            print(f"Final weights:{sample_weights}")
            # self.dataset_len = sum(self.dataset_sizes)
            raw_dataset_len = sum(self.dataset_sizes)
            self.dataset_sample_counts = np.maximum(
                (self.sample_weights * dataset_size_one_epoch).astype(int),
                1
            )
            print(f"Not sampled: Dataset len:{raw_dataset_len}")
            print("Final sampling info:")
            table_data = [
                [self.dataset_names[i], len(self.datasets[i]), f"{self.sample_weights[i]:.4f}"]
                for i in range(len(self.datasets))
            ]
            print(tabulate(table_data, headers=["Dataset", "Samples", "Ratio"], tablefmt="grid"))
            # sample and use NamedSubset to contain dataset_name
            self.id2dataset, self.num_episodes, self.dataset_len = self.build_pretrain_id2dataset(seed=seed)
            self.dataset = None
            self.dataset_len = len(self.id2dataset)
            
        # calculate stats
        self.max_action_dim = cfg.policy.max_action_dim
        self.max_state_dim = cfg.policy.max_state_dim
        all_new_obs_image_keys = ["observation.images.primary", 
                                  "observation.images.secondary", 
                                  "observation.images.wrist"] # follow https://github.com/openvla/openvla/blob/main/prismatic/vla/datasets/rlds/oxe/configs.py
        
        # print(self.datasets[0].meta.stats)
        self.stats = aggregate_stats_with_padding([dataset.meta.stats for dataset in self.datasets], 
                                     max_dim = self.max_action_dim) # Note: I modified this function
        
        print(f"Aggregated stats:{self.stats}")
        # update meta_features
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] - meta features: {meta_features}")
        new_obs_image_keys = []
        for key in self.stats.keys():
            if key in all_new_obs_image_keys:
                new_obs_image_keys.append(key)
        self.new_obs_image_keys = new_obs_image_keys
        img_feats = {}
        # first remove keys contaning images
        old_keys = list(meta_features.keys())
        print("\n\n")
        for key in old_keys:
            # print(key, meta_features[key])
            if meta_features[key]["dtype"] in ["image", "video"]:
                img_feats = meta_features[key]
                del meta_features[key]
        # update the size of image feats
        # print(f"Old image features:{img_feats}")
        img_size = cfg.dataset.image_transforms.img_size
        img_feats["shape"] = (img_size, img_size, 3)
        img_feats["info"]["video.height"] = img_size
        img_feats["info"]["video.width"] = img_size
        # then use the new image keys
        for new_key in new_obs_image_keys:
            meta_features[new_key] = img_feats
        print(f"Unified input features:{meta_features}")
        # finally create the meta class
        self.meta = LeRobotDatasetMetadata.create_with_stats_feats(stats=self.stats, features=meta_features) # Note: I added a class function
        self.meta.repo_id = "Prometheus"
        # print(self.meta.features)
        # self.dataset = None
    
    def build_pretrain_id2dataset(self, seed: int = 1000):
        random.seed(seed)
        id2dataset = []
        episode_count = 0
        for dataset_idx, (dataset, num_samples, dataset_name) in enumerate(
            zip(self.datasets, self.dataset_sample_counts, self.dataset_names)
        ):
            indices = list(range(len(dataset)))
        
            if num_samples <= len(indices):
                # 不放回
                sampled_indices = random.sample(indices, num_samples)
            else:
                # 放回
                sampled_indices = random.choices(indices, k=num_samples)
        
            episode_this_dataset = int(
                dataset.num_episodes * (len(sampled_indices) / len(dataset))
            )
            episode_count += episode_this_dataset
        
            # self.selected_indices.append(sampled_indices)
        
            # ⭐ 构建 id -> (dataset_idx, data_id)
            for data_id in sampled_indices:
                id2dataset.append((dataset_idx, data_id))
        return id2dataset, episode_count, len(id2dataset)
    
    def set_epoch(self, epoch):
        self.id2dataset, self.num_episodes, self.dataset_len = self.build_pretrain_id2dataset(seed=epoch + self.seed)    
    
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

    def __getitem__(self, index):
        # NOTE: Removed print statement that was causing significant performance overhead!
        # Each print in __getitem__ requires I/O operations and in DataLoader with
        # num_workers > 0, prints from workers are serialized, slowing down data loading.
        import time
        t0 = time.perf_counter()
        # if self.is_ft:
        dataset_id, data_id = self.id2dataset[index]
        t1 = time.perf_counter()
        dataset = self.datasets[dataset_id]
        t2 = time.perf_counter()
        item = dataset[data_id]
        t3 = time.perf_counter()
        dataset_name = item["dataset_name"]
        data_config = OXE_DATASET_CONFIGS[dataset_name]
        image_obs_keys = data_config["image_obs_keys"]
        t4 = time.perf_counter()
        data_dict = self._fetch_data_dict(item, image_obs_keys)
        t5 = time.perf_counter()
        # print(
        #     f"[MultiDatasetforDistTraining.__getitem__] "
        #     f"id2dataset lookup: {(t1 - t0) * 1000:.3f}ms, "
        #     f"dataset access: {(t2 - t1) * 1000:.3f}ms, "
        #     f"LeRobotDataset __getitem__: {(t3 - t2) * 1000:.3f}ms, "
        #     f"get config: {(t4 - t3) * 1000:.3f}ms, "
        #     f"_fetch_data_dict: {(t5 - t4) * 1000:.3f}ms, "
        #     f"total: {(t5 - t0) * 1000:.3f}ms"
        # )
        return data_dict
    
    def _fetch_data_dict(self, item, image_obs_keys):
        
        exist_image = None
        key_to_pad = []
        new_keys = []
        for new_key, old_key in image_obs_keys.items():
            new_keys.append(f"observation.images.{new_key}")
            # for interna1, its image key is images.rgb.{old_key} instead of observation.images.{old_key}
            old_img_key = f"observation.images.{old_key}" if f"observation.images.{old_key}" in item else f"images.rgb.{old_key}"
            if old_img_key not in item:
                old_img_key = f"observations.images.{old_key}" # for ms buy data
            if old_key != None:
                
                if isinstance(item[old_img_key], list):
                    if not len(item[old_img_key]):
                        key_to_pad.append(new_key)
                
                item[f"observation.images.{new_key}"] = copy.deepcopy(item[old_img_key])
                exist_image = item[old_img_key]
                if new_key != old_key:
                    del item[old_img_key]
            else:
                # if missing, use zero image
                key_to_pad.append(new_key)
        
        exist_image_valide = False
        if exist_image is not None:
            if isinstance(exist_image, list):
                if len(exist_image) > 0:
                    height, width = exist_image[0].size
                    channel = len(exist_image[0].split())
                    sample_image = Image.fromarray(np.ones((height, width, channel), dtype=np.uint8))
                    exist_image_valide = True
            elif isinstance(exist_image, Image.Image):
                height, width = exist_image.size
                channel = len(exist_image.split())
                sample_image = Image.fromarray(np.ones((height, width, channel), dtype=np.uint8))
                exist_image_valide = True
        
        if not exist_image_valide:
            sample_image = Image.fromarray(np.ones((self.cfg.dataset.default_image_size, self.cfg.dataset.default_image_size, self.cfg.dataset.default_channel_size), dtype=np.uint8))  
        
        for new_key in key_to_pad:
            item[f"observation.images.{new_key}"] = copy.deepcopy(sample_image)
            if new_key == "primary":
                item[f"observation.images.{new_key}"] = [item[f"observation.images.{new_key}"]]
        
        # remove other image keys
        keys = list(item.keys())
        for key in keys:
            if "images" in key and key not in new_keys:
                del item[key]
                
        # add the dataset source
        if "episode_index" in item:
            item["source"] = f"{item['dataset_name']}_episode_id_{item['episode_index']}"
        elif "ep_idx" in item:
            item["source"] = f"{item['dataset_name']}_episode_id_{item['ep_idx']}"
        else:
            item["source"] = f"{item['dataset_name']}_with_unknown_episode_id"
        
        raw_action_dim = item["action"].shape[-1]
        print(raw_action_dim)
        # Pad the action and observation vectors
        item["action"] = self.pad_vector(item["action"], self.max_action_dim)
        item["observation.state"] = self.pad_vector(item["observation.state"], self.max_state_dim)
        
        # Normlize the action and observation vectors
        # if "agi" in item["dataset_name"] or "dual" in item["dataset_name"] or "agilex" in item["dataset_name"]:
        if raw_action_dim > 10:
            xyz_idx = [0, 1, 2, 10, 11, 12]   # 双臂 xyz
        else:
            xyz_idx = [0, 1, 2]               # 单臂 xyz

        # print(t/orch.max(item["action"]), torch.min(item["action"]))
        # action
        mean = self.stats["action"]["mean"].to(item["action"].dtype).to(item["action"].device)
        std = self.stats["action"]["std"].to(item["action"].dtype).to(item["action"].device)
        item["action"][..., xyz_idx] = (item["action"][..., xyz_idx] - mean[xyz_idx]) / (std[xyz_idx] + 1e-8)

        # state
        mean = self.stats["observation.state"]["mean"].to(item["observation.state"].dtype).to(item["observation.state"].device)
        std = self.stats["observation.state"]["std"].to(item["observation.state"].dtype).to(item["observation.state"].device)
        item["observation.state"][..., xyz_idx] = (
            item["observation.state"][..., xyz_idx] - mean[xyz_idx]
        ) / (std[xyz_idx] + 1e-8)
        
        item["timestamp"] = item["timestamp"].unsqueeze(0) # make it (1,) for later processing
        # print(item["timestamp"].shape)
        # pil_image = Image.fromarray(item["observation.images.primary"][0])
        # print(item["observation.images.primary"].shape)
        return_dict = {
            "sample_rate": item["fps"],
            "action": item["action"],
            "observation.images.primary": item["observation.images.primary"],
            "observation.state": item["observation.state"],
            "source": item["source"],
            "timestamp": item["timestamp"],
        }
        return return_dict
    
    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        # return len(self.dataset) if self.dataset is not None else self.meta.total_frames
        return self.dataset_len if self.dataset_len is not None else self.meta.total_frames

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.dataset is not None:
            return self.dataset.features
        else:
            return get_hf_features_from_features(self.features)
    
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
        if "action" in key and cfg.action_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.action_delta_indices]
        if key.startswith("observation.") and cfg.observation_delta_indices is not None:
            delta_timestamps[key] = [i / ds_meta.fps for i in cfg.observation_delta_indices]

    if len(delta_timestamps) == 0:
        delta_timestamps = None

    return delta_timestamps
