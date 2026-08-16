#!/usr/bin/env python
"""
Visualize VAE features from RobotCLIP model.

This script loads the RobotCLIP model, reads libero dataset, 
extracts vae_feature and visualizes it alongside the original image.

Usage:
    conda activate lerobot_v2
    python scripts/visualize_vae_feature.py
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.common.policies.ace.modeling_robo_clip import RobotCLIP
from lerobot.common.policies.ace.configuration_robo_clip import RobotCLIPConfig
from lerobot.common.datasets.lerobot_dataset_for_ace import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
    MultiDatasetforDistTraining,
    resolve_delta_timestamps,
)
from lerobot.common.datasets.mixtures import OXE_NAMED_MIXTURES
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.oxe_configs import OXE_DATASET_CONFIGS


def load_model(weight_path: str, device: str = "cuda"):
    """Load RobotCLIP model from weight path."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading model from {weight_path}")
    
    # Create config
    config = RobotCLIPConfig(
        action_dim=7,
        chunk_size=16,
        group_size=4,
        hidden_dim=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        output_dim=768,
        vision_model_name="/Data/lzl/huggingface/siglip2-base-patch16-224",
        projection_dim=768,
        max_action_dim=32,
        max_state_dim=32,
    )
    
    # Create model
    model = RobotCLIP(config)
    model = model.to(device)
    model = model.to(torch.bfloat16)
    
    # Load weights (deepspeed format)
    weight_file = os.path.join(weight_path, "mp_rank_00_model_states.pt")
    if os.path.exists(weight_file):
        state_dict = torch.load(weight_file, map_location=device)
        # Extract module from deepspeed wrapper if needed
        if "module" in state_dict:
            state_dict = state_dict["module"]
        model.load_state_dict(state_dict, strict=False)
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Model weights loaded successfully")
    else:
        print(f"Warning: Weight file not found at {weight_file}, using random weights")
    
    model.eval()
    return model, config


def create_image_transforms(img_size: int = 224):
    """Create image transforms."""
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.configs.types import ImageTransformsConfig
    
    # Create a minimal config for image transforms
    img_transform_config = ImageTransformsConfig(
        enable=False,
        img_size=img_size,
    )
    return ImageTransforms(img_transform_config)


def load_libero_dataset(parent_dir: str, data_mix: str = "libero", vla2root_json: str = "vla2root.json"):
    """Load libero dataset."""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading {data_mix} dataset from {parent_dir}")
    
    # Load vla2root mapping
    with open(vla2root_json, "r") as f:
        vla2data_root = json.load(f)
    
    # Get dataset mixture spec
    mixture_spec = OXE_NAMED_MIXTURES[data_mix]
    included_datasets = [d_name for d_name, d_weight in mixture_spec]
    
    print(f"Datasets in mixture: {included_datasets}")
    
    # Load individual datasets
    datasets = []
    dataset_names = []
    
    for dataset_name in included_datasets:
        if dataset_name in vla2data_root:
            data_root = os.path.join(parent_dir, vla2data_root[dataset_name])
            print(f"Loading {dataset_name} from {data_root}")
            
            repo_id = f"bulldog-{dataset_name}"
            try:
                ds_meta = LeRobotDatasetMetadata(repo_id, root=data_root)
                
                # Create simple config for delta_timestamps
                from dataclasses import dataclass
                @dataclass
                class SimpleConfig:
                    action_delta_indices: list = None
                    observation_delta_indices: list = None
                    reward_delta_indices: list = None
                
                simple_cfg = SimpleConfig(
                    action_delta_indices=list(range(16)),
                )
                
                delta_timestamps = resolve_delta_timestamps(simple_cfg, ds_meta)
                
                dataset = LeRobotDataset(
                    repo_id,
                    root=data_root,
                    delta_timestamps=delta_timestamps,
                    image_transforms=None,
                    video_backend="pyav",
                    dataset_name=dataset_name,
                )
                datasets.append(dataset)
                dataset_names.append(dataset_name)
                print(f"Loaded {dataset_name} with {len(dataset)} frames")
            except Exception as e:
                print(f"Failed to load {dataset_name}: {e}")
        else:
            print(f"Dataset {dataset_name} not found in vla2root.json")
    
    return datasets, dataset_names


def visualize_vae_feature(model, image: Image.Image, device: str = "cuda"):
    """Extract and visualize VAE feature from model."""
    # Convert image to PIL format if needed
    if isinstance(image, torch.Tensor):
        if image.dim() == 4:
            image = image.squeeze(0)
        if image.shape[0] == 3:  # CHW
            image = image.permute(1, 2, 0)
        image = Image.fromarray((image.cpu().numpy() * 255).astype("uint8"))
    
    # Get vision model output
    with torch.no_grad():
        vision_output = model.vision_model([image])
        vae_feature = vision_output["vae_feature"]  # [B, C, H, W]
    
    # vae_feature shape: [1, 16, 28, 28]
    vae_feature_np = vae_feature.float().cpu().numpy()[0]  # [C, H, W]
    
    return vae_feature_np, image


def create_visualization(original_image: Image.Image, vae_feature: np.ndarray, save_path: str, index: int):
    """Create visualization comparing original image and VAE feature."""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # VAE feature channels (show first 4 channels)
    num_channels = vae_feature.shape[0]
    for i in range(4):
        if i < num_channels:
            # Normalize feature for visualization
            channel_feature = vae_feature[i]
            channel_feature = (channel_feature - channel_feature.min()) / (channel_feature.max() - channel_feature.min() + 1e-8)
            axes[i + 1].imshow(channel_feature, cmap="viridis")
            axes[i + 1].set_title(f"VAE Channel {i}")
        else:
            axes[i + 1].axis("off")
        axes[i + 1].axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"vae_feature_{index}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    
    # Also create a summary visualization with all channels
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for i in range(16):
        row = i // 4
        col = i % 4
        if i < num_channels:
            channel_feature = vae_feature[i]
            channel_feature = (channel_feature - channel_feature.min()) / (channel_feature.max() - channel_feature.min() + 1e-8)
            axes[row, col].imshow(channel_feature, cmap="viridis")
            axes[row, col].set_title(f"Ch {i}")
        axes[row, col].axis("off")
    
    plt.suptitle(f"All VAE Feature Channels (shape: {vae_feature.shape})", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"vae_feature_all_channels_{index}.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    """Main function to visualize VAE features."""
    # Configuration
    weight_path = "/Data/lzl/ace_weights/0520_ace"
    parent_dir = "/Data/lerobot_data_ort6d"
    data_mix = "libero"
    vla2root_json = "vla2root.json"
    output_dir = "scripts/vae_feature_vis"
    num_samples = 50  # Number of samples to visualize
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    model, config = load_model(weight_path, device)
    
    # Load dataset
    datasets, dataset_names = load_libero_dataset(parent_dir, data_mix, vla2root_json)
    
    if len(datasets) == 0:
        print("No datasets loaded!")
        return
    
    # Sample from each dataset
    sample_count = 0
    for ds_idx, (dataset, dataset_name) in enumerate(zip(datasets, dataset_names)):
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Processing {dataset_name}")
        
        # Sample indices
        total_frames = len(dataset)
        sample_indices = np.random.choice(total_frames, min(num_samples, total_frames), replace=False)
        
        for idx in sample_indices:
            try:
                # Get data item
                item = dataset[int(idx)]
                # print(item.keys())
                # Get primary image
                if "observation.images.image" in item:
                    # print("  Found primary image key")
                    images = item["observation.images.image"]
                    if isinstance(images, list):
                        image = images[0]
                    else:
                        image = images
                else:
                    # Try to find any image key
                    for key in item.keys():
                        if "observation.images" in key:
                            images = item[key]
                            if isinstance(images, list):
                                image = images[0]
                            else:
                                image = images
                            break
                
                if isinstance(image, torch.Tensor):
                    if image.dim() == 4:
                        image = image.squeeze(0)
                    if image.shape[0] == 3:  # CHW format
                        image = image.permute(1, 2, 0)
                    # Convert to PIL Image
                    if image.max() <= 1.0:
                        image = Image.fromarray((image.cpu().numpy() * 255).astype("uint8"))
                    else:
                        image = Image.fromarray(image.cpu().numpy().astype("uint8"))
                
                # Extract VAE feature
                vae_feature, original_image = visualize_vae_feature(model, image, device)
                
                # Create visualization
                create_visualization(original_image, vae_feature, output_dir, sample_count)
                
                print(f"  [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved visualization for sample {sample_count} from {dataset_name}")
                
                sample_count += 1
                if sample_count >= num_samples:
                    break
                    
            except Exception as e:
                print(f"  Error processing sample {idx}: {e}")
                continue
        
        if sample_count >= num_samples:
            break
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Completed! Total samples visualized: {sample_count}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()