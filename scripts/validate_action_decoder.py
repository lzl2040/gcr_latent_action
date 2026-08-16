#!/usr/bin/env python
"""
Validation script for action encoder and decoder.

This script loads the libero dataset, runs actions through the action_encoder and decoder,
and compares the reconstructed actions with the original actions.

Usage:
    python scripts/validate_action_decoder.py
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lerobot.common.datasets.lerobot_dataset_for_ace import (
    MultiDatasetforDistTraining,
    extra_collate_fn,
)
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.oxe_configs import OXE_DATASET_CONFIGS
from lerobot.common.policies.ace.modeling_robo_clip import RobotCLIP
from lerobot.common.policies.ace.configuration_robo_clip import RobotCLIPConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs import parser


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pretrained_weights(model, pretrained_path):
    """Load pretrained weights for the action_decoder."""
    print(f"Loading pretrained weights from {pretrained_path}")
    
    if os.path.isdir(pretrained_path):
        # If it's a directory, look for model states file
        ckpt_files = list(Path(pretrained_path).glob("mp_rank_*_model_states.pt"))
        if ckpt_files:
            ckpt_path = ckpt_files[0]
        else:
            ckpt_files = list(Path(pretrained_path).glob("*.pt"))
            if ckpt_files:
                ckpt_path = ckpt_files[0]
            else:
                raise FileNotFoundError(f"No checkpoint files found in {pretrained_path}")
    else:
        ckpt_path = pretrained_path
    
    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if "module" in checkpoint:
            state_dict = checkpoint["module"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    print(f"Missing keys: {len(missing_keys)}")
    if missing_keys:
        print(f"Sample missing keys: {missing_keys[:5]}")
    print(f"Unexpected keys: {len(unexpected_keys)}")
    if unexpected_keys:
        print(f"Sample unexpected keys: {unexpected_keys[:5]}")
    
    return model


def compute_action_metrics(pred_actions, gt_actions):
    """
    Compute various metrics between predicted and ground truth actions.
    
    Args:
        pred_actions: torch.Tensor of shape (B, chunk_size, action_dim)
        gt_actions: torch.Tensor of shape (B, chunk_size, action_dim)
    
    Returns:
        dict of metrics
    """
    # MSE per step
    print(pred_actions[0, 0, :5], "gt:", "", gt_actions[0, 0, :5])
    mse_per_step = torch.mean((pred_actions - gt_actions) ** 2, dim=-1)  # (B, chunk_size)
    
    # Mean MSE across all dimensions
    mse_total = torch.mean((pred_actions - gt_actions) ** 2).item()
    
    # MAE
    mae_total = torch.mean(torch.abs(pred_actions - gt_actions)).item()
    
    # Per-step MSE
    mse_by_step = torch.mean(mse_per_step, dim=0).float()  # (chunk_size,) - convert to float32 for numpy
    
    # L2 distance
    l2_dist = torch.norm(pred_actions - gt_actions, dim=-1)  # (B, chunk_size)
    mean_l2 = torch.mean(l2_dist).item()
    
    # Cosine similarity
    pred_flat = pred_actions.view(-1, pred_actions.shape[-1])
    gt_flat = gt_actions.view(-1, gt_actions.shape[-1])
    cosine_sim = torch.nn.functional.cosine_similarity(pred_flat, gt_flat, dim=-1)
    mean_cosine_sim = torch.mean(cosine_sim).item()
    
    return {
        "mse_total": mse_total,
        "mae_total": mae_total,
        "mean_l2": mean_l2,
        "mean_cosine_sim": mean_cosine_sim,
        "mse_by_step": mse_by_step.cpu().numpy(),
    }


@torch.no_grad()
def validate_action_decoder(cfg, num_samples=2000, batch_size=8, device="cuda"):
    """
    Main validation function.
    
    Args:
        cfg: TrainPipelineConfig
        num_samples: Number of samples to validate
        batch_size: Batch size for validation
        device: Device to use
    """
    print("=" * 60)
    print("Action Decoder Validation Script")
    print("=" * 60)
    
    # Setup image transforms
    image_transforms = ImageTransforms(cfg.dataset.image_transforms)
    wrist_image_transforms = ImageTransforms(cfg.dataset.wrist_image_transforms)
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset (data_mix=libero)...")
    
    # Load dataset
    dataset = MultiDatasetforDistTraining(
        cfg=cfg,
        image_transforms=image_transforms,
        wrist_image_transforms=wrist_image_transforms,
        seed=cfg.seed,
        data_mix="libero",
        vla2root_json="vla2root.json",
        dataset_size_one_epoch=num_samples,
    )
    
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Create dataloader
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=extra_collate_fn,
        pin_memory=True,
    )
    
    # Create model config
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Creating model...")
    model_config = RobotCLIPConfig(
        action_dim=cfg.policy.action_dim,
        chunk_size=cfg.policy.chunk_size,
        group_size=cfg.policy.group_size,
        hidden_dim=cfg.policy.hidden_dim,
        num_attention_heads=cfg.policy.num_attention_heads,
        num_hidden_layers=cfg.policy.num_hidden_layers,
        output_dim=cfg.policy.output_dim,
        max_action_dim=cfg.policy.max_action_dim,
        frozen_ace=True,  # Enable action_decoder
        vision_model_name=cfg.policy.vision_model_name,
    )
    
    # Create model
    model = RobotCLIP(model_config, dataset_stats=dataset.stats)
    
    # Load pretrained weights
    pretrained_path = cfg.policy.pretrained_path
    if pretrained_path and os.path.exists(pretrained_path):
        model = load_pretrained_weights(model, pretrained_path)
    else:
        print(f"Warning: Pretrained path not found: {pretrained_path}")
        print("Using randomly initialized weights...")
    
    # Move to device and convert to bfloat16
    model = model.to(device)
    model = model.to(torch.bfloat16)
    model.eval()
    
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting validation on {min(num_samples, len(dataset))} samples...")
    
    # Metrics accumulation
    all_mse = []
    all_mae = []
    all_l2 = []
    all_cosine_sim = []
    mse_by_step_all = []
    
    num_processed = 0
    
    for batch in tqdm(dataloader, total=min(num_samples // batch_size + 1, len(dataloader))):
        if num_processed >= num_samples:
            break
        
        # Move batch to device
        actions = batch["action"].to(device, dtype=torch.bfloat16)
        sample_rate = batch.get("sample_rate", 0)
        if isinstance(sample_rate, torch.Tensor):
            sample_rate = sample_rate.to(device)
        
        # Forward pass through action encoder and decoder
        action_output = model.action_encoder(actions, sample_rate)
        reconstructed_actions = action_output["reconstructed_actions"]
        recon_loss = action_output["recon_loss"]
        
        # Compute metrics
        metrics = compute_action_metrics(reconstructed_actions, actions)
        
        all_mse.append(metrics["mse_total"])
        all_mae.append(metrics["mae_total"])
        all_l2.append(metrics["mean_l2"])
        all_cosine_sim.append(metrics["mean_cosine_sim"])
        mse_by_step_all.append(metrics["mse_by_step"])
        
        num_processed += actions.shape[0]
    
    # Compute final statistics
    final_metrics = {
        "num_samples": num_processed,
        "mse_mean": np.mean(all_mse),
        "mse_std": np.std(all_mse),
        "mae_mean": np.mean(all_mae),
        "mae_std": np.std(all_mae),
        "l2_mean": np.mean(all_l2),
        "l2_std": np.std(all_l2),
        "cosine_sim_mean": np.mean(all_cosine_sim),
        "cosine_sim_std": np.std(all_cosine_sim),
    }
    
    # Compute average MSE by step
    mse_by_step_avg = np.mean(mse_by_step_all, axis=0)
    
    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    print(f"Number of samples processed: {final_metrics['num_samples']}")
    print("-" * 40)
    print(f"MSE (Mean Squared Error):")
    print(f"  Mean: {final_metrics['mse_mean']:.6f}")
    print(f"  Std:  {final_metrics['mse_std']:.6f}")
    print(f"\nMAE (Mean Absolute Error):")
    print(f"  Mean: {final_metrics['mae_mean']:.6f}")
    print(f"  Std:  {final_metrics['mae_std']:.6f}")
    print(f"\nL2 Distance:")
    print(f"  Mean: {final_metrics['l2_mean']:.6f}")
    print(f"  Std:  {final_metrics['l2_std']:.6f}")
    print(f"\nCosine Similarity:")
    print(f"  Mean: {final_metrics['cosine_sim_mean']:.6f}")
    print(f"  Std:  {final_metrics['cosine_sim_std']:.6f}")
    print("-" * 40)
    print("\nMSE by action step:")
    for i, mse in enumerate(mse_by_step_avg):
        print(f"  Step {i:2d}: {mse:.6f}")
    print("=" * 60)
    
    # Save results to JSON
    results_path = Path("scripts/action_decoder_validation_results.json")
    with open(results_path, "w") as f:
        json.dump({
            "final_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v 
                              for k, v in final_metrics.items()},
            "mse_by_step": [float(v) for v in mse_by_step_avg],
            "config": {
                "chunk_size": cfg.policy.chunk_size,
                "action_dim": cfg.policy.action_dim,
                "hidden_dim": cfg.policy.hidden_dim,
                "pretrained_path": pretrained_path,
            }
        }, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return final_metrics


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    """Main entry point."""
    cfg.validate()
    
    # Set default paths if not specified
    if cfg.dataset.parent_dir is None:
        cfg.dataset.parent_dir = "/Data/lerobot_data_ort6d"
    
    if cfg.policy.pretrained_path is None or cfg.policy.pretrained_path == "":
        cfg.policy.pretrained_path = "/Data/lzl/ace_weights/action_decoder_weights"
    
    # Run validation
    validate_action_decoder(
        cfg=cfg,
        num_samples=2000,
        batch_size=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )


if __name__ == "__main__":
    main()