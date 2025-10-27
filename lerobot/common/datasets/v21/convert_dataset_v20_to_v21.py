"""
This script will help you convert any LeRobot dataset already pushed to the hub from codebase version 2.0 to
2.1. It will:

- Generate per-episodes stats and writes them in `episodes_stats.jsonl`
- Check consistency between these new stats and the old ones.
- Remove the deprecated `stats.json`.
- Update codebase_version in `info.json`.
- Push this new version to the hub on the 'main' branch and tags it with "v2.1".

Usage:

```bash
python lerobot/common/datasets/v21/convert_dataset_v20_to_v21.py \
    --repo-id=aliberts/koch_tutorial
```

"""

import argparse
import logging
import concurrent.futures

from huggingface_hub import HfApi

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION, LeRobotDataset
from lerobot.common.datasets.compute_stats import DEFAULT_QUANTILES, aggregate_stats, get_feature_stats
from lerobot.common.datasets.utils import EPISODES_STATS_PATH, STATS_PATH, load_stats, write_info
from lerobot.common.datasets.v21.convert_stats import check_aggregate_stats, convert_stats
from lerobot.common.datasets.utils import write_stats

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader


V20 = "v2.0"
V21 = "v2.1"
V30 = "v3.0"

class SuppressWarnings:
    def __enter__(self):
        self.previous_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.getLogger().setLevel(self.previous_level)

# def process_single_episode(dataset: LeRobotDataset, episode_idx: int, episode_data_index: dict) -> dict:
def process_single_episode(data_iter, dataset, episode_idx: int, episode_data_index: dict) -> dict:
    """Process a single episode and return its statistics.

    Args:
        dataset: The LeRobot dataset
        episode_idx: Index of the episode to process

    Returns:
        Dictionary containing episode statistics
    """
    logging.info(f"Computing stats for episode {episode_idx}")

    # start_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    # end_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
    start_idx = episode_data_index["from"][episode_idx]
    end_idx = episode_data_index["to"][episode_idx]
    ep_stats = {}

    collected_data: dict[str, list] = {}
    for idx in range(start_idx, end_idx):
        # item = dataset[idx]
        item = next(data_iter)[0]
    #     print(item)
        for key, value in item.items():
            if key not in dataset.features:
                continue

            if key not in collected_data:
                collected_data[key] = []
            collected_data[key].append(value)

    ep_stats = {}
    for key, data_list in collected_data.items():
        if dataset.features[key]["dtype"] == "string":
            continue

        data = torch.stack(data_list).cpu().numpy()
        if dataset.features[key]["dtype"] in ["image", "video"]:
            if data.dtype == np.uint8:
                data = data.astype(np.float32) / 255.0

            axes_to_reduce = (0, 2, 3)
            keepdims = True
        else:
            axes_to_reduce = 0
            keepdims = data.ndim == 1

        ep_stats[key] = get_feature_stats(
            data, axis=axes_to_reduce, keepdims=keepdims, quantile_list=DEFAULT_QUANTILES
        )

        if dataset.features[key]["dtype"] in ["image", "video"]:
            ep_stats[key] = {
                k: v if k == "count" else np.squeeze(v, axis=0) for k, v in ep_stats[key].items()
            }

    return ep_stats

def compute_quantile_stats_for_dataset(dataset: LeRobotDataset) -> dict[str, dict]:
    """Compute quantile statistics for all episodes in the dataset.

    Args:
        dataset: The LeRobot dataset to compute statistics for

    Returns:
        Dictionary containing aggregated statistics with quantiles

    Note:
        Video decoding operations are not thread-safe, so we process episodes sequentially
        when video keys are present. For datasets without videos, we use parallel processing
        with ThreadPoolExecutor for better performance.
    """
    logging.info(f"Computing quantile statistics for dataset with {dataset.num_episodes} episodes")

    episode_stats_list = []
    has_videos = len(dataset.meta.video_keys) > 0
    from lerobot.common.datasets.utils import get_episode_data_index
    episode_data_index = get_episode_data_index(dataset.meta.episodes)
    loader = DataLoader(dataset, batch_size=1, num_workers=8, shuffle=False, collate_fn=lambda x: x)

    episode_iter = iter(loader)

    if has_videos:
        logging.info("Dataset contains video keys - using sequential processing for thread safety")
        for episode_idx in tqdm(range(dataset.num_episodes), desc="Processing episodes"):
            ep_stats = process_single_episode(episode_iter, dataset, episode_idx, episode_data_index)
            episode_stats_list.append(ep_stats)
            # ep_stats = process_single_episode(dataset, episode_idx, episode_data_index)
            # episode_stats_list.append(ep_stats)
    else:
        logging.info("Dataset has no video keys - using parallel processing for better performance")
        max_workers = min(dataset.num_episodes, 16)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_episode = {
                executor.submit(process_single_episode, dataset, episode_idx): episode_idx
                for episode_idx in range(dataset.num_episodes)
            }

            episode_results = {}
            with tqdm(total=dataset.num_episodes, desc="Processing episodes") as pbar:
                for future in concurrent.futures.as_completed(future_to_episode):
                    episode_idx = future_to_episode[future]
                    ep_stats = future.result()
                    episode_results[episode_idx] = ep_stats
                    pbar.update(1)

        for episode_idx in range(dataset.num_episodes):
            if episode_idx in episode_results:
                episode_stats_list.append(episode_results[episode_idx])

    if not episode_stats_list:
        raise ValueError("No episode data found for computing statistics")

    logging.info(f"Aggregating statistics from {len(episode_stats_list)} episodes")
    return aggregate_stats(episode_stats_list)

def convert_dataset(
    repo_id: str,
    branch: str | None = None,
    num_workers: int = 4,
):
    with SuppressWarnings():
        dataset = LeRobotDataset(repo_id, revision=V20, force_cache_sync=True)

    if (dataset.root / EPISODES_STATS_PATH).is_file():
        (dataset.root / EPISODES_STATS_PATH).unlink()

    convert_stats(dataset, num_workers=num_workers)
    ref_stats = load_stats(dataset.root)
    check_aggregate_stats(dataset, ref_stats)

    dataset.meta.info["codebase_version"] = CODEBASE_VERSION
    write_info(dataset.meta.info, dataset.root)

    dataset.push_to_hub(branch=branch, tag_version=False, allow_patterns="meta/")

    # delete old stats.json file
    if (dataset.root / STATS_PATH).is_file:
        (dataset.root / STATS_PATH).unlink()

    hub_api = HfApi()
    if hub_api.file_exists(
        repo_id=dataset.repo_id, filename=STATS_PATH, revision=branch, repo_type="dataset"
    ):
        hub_api.delete_file(
            path_in_repo=STATS_PATH, repo_id=dataset.repo_id, revision=branch, repo_type="dataset"
        )

    hub_api.create_tag(repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")


def convert_dataset_with_data_mix(
        data_mix: str = "simpler_bridge",
        num_workers: int = 4,):
    from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
    import json
    import os
    # vla2root_json = "/home/v-zuoleili/Project/gcr_latent_action/vla2root.json"
    # parent_dir = "/home/v-zuoleili/Data/lerobot_data"
    vla2root_json = "vla2root.json"
    parent_dir = "/mnt/wangxiaofa/robot_dataset/lerobot-format"
    mixture_spec = OXE_NAMED_MIXTURES[data_mix]
    included_datasets, sample_weights = [], []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.append(d_name)
        sample_weights.append(d_weight)
    
    datasets = []
    with open(vla2root_json, "r") as f:
        vla2data_root = json.load(f)
    for dataset_name in included_datasets:
        
        if dataset_name in vla2data_root.keys():
            data_root = vla2data_root[dataset_name]
            data_root = os.path.join(parent_dir, data_root)

            print(f"Load data from {data_root}")
            repo_id = f"bulldog-{dataset_name}" # any
            dataset = LeRobotDataset(
                repo_id, 
                root=data_root,
                # revision=V20, 
                # force_cache_sync=True
            )
            # print(f"Processing {dataset_name}")
            # if (dataset.root / EPISODES_STATS_PATH).is_file():
            #     print(f"{dataset_name} has {EPISODES_STATS_PATH}")
            # else:
            #     convert_stats(dataset, num_workers=num_workers)
            #     ref_stats = load_stats(dataset.root)
            #     check_aggregate_stats(dataset, ref_stats)

            #     dataset.meta.info["codebase_version"] = V21
            #     write_info(dataset.meta.info, dataset.root)

            print("Updating quantile_stats_for_dataset")
            # update quantile_stats_for_dataset
            dataset = LeRobotDataset(
                repo_id, 
                root=data_root,
            )
            new_stats = compute_quantile_stats_for_dataset(dataset)
            logging.info("Updating dataset metadata with new quantile statistics")
            dataset.meta.stats = new_stats

            write_stats(new_stats, dataset.meta.root)
            dataset.meta.info["codebase_version"] = V30
            write_info(dataset.meta.info, dataset.root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--repo-id",
    #     type=str,
    #     required=True,
    #     help="Repository identifier on Hugging Face: a community or a user name `/` the name of the dataset "
    #     "(e.g. `lerobot/pusht`, `cadene/aloha_sim_insertion_human`).",
    # )
    # parser.add_argument(
    #     "--branch",
    #     type=str,
    #     default=None,
    #     help="Repo branch to push your dataset. Defaults to the main branch.",
    # )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for parallelizing stats compute. Defaults to 4.",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default="simpler_bridge",
        help="data.",
    )

    args = parser.parse_args()
    # convert_dataset(**vars(args))
    convert_dataset_with_data_mix(**vars(args))
