"""Validate the unified latent-action decoder on actions and future images."""

import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

# Running ``python scripts/...`` puts the scripts directory, rather than the
# repository root, first on sys.path. Prefer this checkout over any editable
# LeRobot installation from another workspace.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

# vt_dataset imports qwen_vl_utils eagerly, although the InternVL validation
# path below never calls it. Keep that optional dependency from blocking this
# script in the lerobot_latent environment.
try:
    import qwen_vl_utils  # noqa: F401
except ModuleNotFoundError:
    qwen_vl_utils_stub = types.ModuleType("qwen_vl_utils")

    def _missing_qwen_vl_utils(*args, **kwargs):
        raise ModuleNotFoundError(
            "qwen_vl_utils is required only when using the Qwen-VL data path"
        )

    qwen_vl_utils_stub.process_vision_info = _missing_qwen_vl_utils
    sys.modules["qwen_vl_utils"] = qwen_vl_utils_stub

from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.vt_dataset import MultiDatasetforDistTraining, extra_collate_fn
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


DEFAULT_CHECKPOINT = Path(
    "/Data/lzl/latent_action/0124_pretrain_latent_unfied_decoder/step80000.pt"
)
DEFAULT_ACTION_STATS = REPO_ROOT / "lerobot/stats/oxe_magic_soup_plus_stats.json"


@dataclass
class UnifiedValidationConfig(TrainPipelineConfig):
    validation_checkpoint: str = str(DEFAULT_CHECKPOINT)
    action_stats_path: str = str(DEFAULT_ACTION_STATS)
    validation_start_batch: int = 0
    num_validation_batches: int = 1000
    num_inference_steps: int = 20
    action_dims: int = 7
    validation_output_dir: str = "unified_validation_step80000"
    save_images: bool = True
    max_saved_images: int = 200


def compute_ssim(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute RGB SSIM with an 11x11 Gaussian window and a [0, 1] data range."""
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if predicted.ndim == 2:
        predicted = predicted[..., None]
    if target.ndim == 2:
        target = target[..., None]
    if predicted.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"Predicted and target images have different channel counts: "
            f"{predicted.shape} vs {target.shape}"
        )
    if predicted.shape[:2] != target.shape[:2]:
        predicted = cv2.resize(
            predicted,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        if predicted.ndim == 2:
            predicted = predicted[..., None]

    predicted = np.clip(predicted, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    height, width = target.shape[:2]
    window_size = min(11, height, width)
    if window_size % 2 == 0:
        window_size -= 1
    if window_size < 3:
        raise ValueError(f"Images are too small for SSIM: {target.shape}")

    sigma = 1.5 * window_size / 11
    kernel = (window_size, window_size)
    mu_pred = cv2.GaussianBlur(predicted, kernel, sigma)
    mu_target = cv2.GaussianBlur(target, kernel, sigma)

    mu_pred_sq = mu_pred * mu_pred
    mu_target_sq = mu_target * mu_target
    mu_cross = mu_pred * mu_target
    sigma_pred_sq = (
        cv2.GaussianBlur(predicted * predicted, kernel, sigma) - mu_pred_sq
    )
    sigma_target_sq = (
        cv2.GaussianBlur(target * target, kernel, sigma) - mu_target_sq
    )
    sigma_cross = (
        cv2.GaussianBlur(predicted * target, kernel, sigma) - mu_cross
    )

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2 * mu_cross + c1) * (2 * sigma_cross + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (
        sigma_pred_sq + sigma_target_sq + c2
    )
    ssim_map = numerator / np.maximum(denominator, 1e-12)

    border = window_size // 2
    if height > 2 * border and width > 2 * border:
        ssim_map = ssim_map[border:-border, border:-border]
    return float(np.mean(ssim_map, dtype=np.float64))


def compute_psnr(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute RGB PSNR in dB with a [0, 1] data range."""
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if predicted.ndim == 2:
        predicted = predicted[..., None]
    if target.ndim == 2:
        target = target[..., None]
    if predicted.shape[-1] != target.shape[-1]:
        raise ValueError(
            f"Predicted and target images have different channel counts: "
            f"{predicted.shape} vs {target.shape}"
        )
    if predicted.shape[:2] != target.shape[:2]:
        predicted = cv2.resize(
            predicted,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        if predicted.ndim == 2:
            predicted = predicted[..., None]

    predicted = np.clip(predicted, 0.0, 1.0)
    target = np.clip(target, 0.0, 1.0)
    mse = float(
        np.mean(
            (predicted.astype(np.float64) - target.astype(np.float64)) ** 2,
            dtype=np.float64,
        )
    )
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10(1.0 / mse))


def load_action_stats(path: Path, action_dims: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Action statistics file does not exist: {path}")
    with path.open("r", encoding="utf-8") as stats_file:
        stats = json.load(stats_file)

    mean = torch.as_tensor(stats["action"]["mean"], dtype=torch.float32)
    std = torch.as_tensor(stats["action"]["std"], dtype=torch.float32)
    if mean.numel() < action_dims or std.numel() < action_dims:
        raise ValueError(
            f"Action stats only have {min(mean.numel(), std.numel())} dimensions, "
            f"but action_dims={action_dims}"
        )
    return mean, std


def move_tensors_to_device(batch: dict, device: torch.device) -> dict:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=True)
    return batch


def get_valid_action_mask(batch: dict, batch_size: int, num_steps: int) -> torch.Tensor:
    action_is_pad = batch.get("action_is_pad")
    if action_is_pad is None:
        return torch.ones(batch_size, num_steps, dtype=torch.bool)

    valid = ~action_is_pad.detach().cpu().bool()
    while valid.ndim > 2 and valid.shape[-1] == 1:
        valid = valid.squeeze(-1)
    if valid.ndim != 2:
        valid = valid.reshape(batch_size, -1)
    return valid[:, :num_steps]


def prepare_rgb_images(images) -> np.ndarray:
    images = np.asarray(images)
    if images.ndim == 3:
        images = images[None]
    if images.ndim != 4:
        raise ValueError(f"Expected a batch of images, got shape {images.shape}")
    if images.shape[1] in (1, 3, 4) and images.shape[-1] not in (1, 3, 4):
        images = images.transpose(0, 2, 3, 1)
    images = images.astype(np.float32)
    if images.max(initial=0.0) > 1.0:
        images /= 255.0
    return np.clip(images[..., :3], 0.0, 1.0)


def save_comparison(
    predicted: np.ndarray,
    target: np.ndarray,
    path: Path,
    gap: int = 20,
) -> None:
    if predicted.shape[:2] != target.shape[:2]:
        predicted = cv2.resize(
            predicted,
            (target.shape[1], target.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    predicted_u8 = np.rint(np.clip(predicted, 0.0, 1.0) * 255).astype(np.uint8)
    target_u8 = np.rint(np.clip(target, 0.0, 1.0) * 255).astype(np.uint8)
    separator = np.full(
        (target_u8.shape[0], gap, target_u8.shape[2]),
        255,
        dtype=np.uint8,
    )
    comparison_rgb = np.concatenate([predicted_u8, separator, target_u8], axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(comparison_rgb, cv2.COLOR_RGB2BGR))


@parser.wrap()
def validate(cfg: UnifiedValidationConfig) -> None:
    cfg.validate()
    if cfg.seed is not None:
        set_seed(cfg.seed)
    if cfg.num_validation_batches <= 0:
        raise ValueError("num_validation_batches must be positive")
    if cfg.validation_start_batch < 0:
        raise ValueError("validation_start_batch must be non-negative")
    if cfg.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be positive")
    if cfg.action_dims <= 0:
        raise ValueError("action_dims must be positive")

    checkpoint_path = Path(cfg.validation_checkpoint).expanduser().resolve()
    action_stats_path = Path(cfg.action_stats_path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("Unified validation requires a CUDA GPU")
    device = torch.device(cfg.device or "cuda:0")
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda:0")

    output_dir = Path(cfg.validation_output_dir).expanduser().resolve()
    image_output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    if cfg.save_images:
        image_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Action stats: {action_stats_path}")
    print(f"Device: {device}")
    print(f"Validation output: {output_dir}")

    image_transforms = ImageTransforms(cfg.dataset.image_transforms)
    wrist_image_transforms = ImageTransforms(cfg.dataset.wrist_image_transforms)
    dataset = MultiDatasetforDistTraining(
        cfg=cfg,
        image_transforms=image_transforms,
        wrist_image_transforms=wrist_image_transforms,
        seed=cfg.seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
        is_train=False,
    )
    start_sample = cfg.validation_start_batch * cfg.batch_size
    end_sample = min(
        start_sample + cfg.num_validation_batches * cfg.batch_size,
        len(dataset),
    )
    if start_sample >= end_sample:
        raise ValueError(
            f"Validation range starts at sample {start_sample}, but the dataset "
            f"only contains {len(dataset)} samples"
        )
    validation_dataset = Subset(dataset, range(start_sample, end_sample))
    dataloader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=extra_collate_fn,
        pin_memory=True,
    )

    cfg.policy.use_unified_decoder = True
    cfg.policy.pretrained_path = str(checkpoint_path)
    cfg.policy.set_token_idx(dataset.cp_act_token_idx, dataset.cp_sc_token_idx)
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=str(checkpoint_path),
    )
    policy = policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    model_action_mean, model_action_std = load_action_stats(
        action_stats_path, cfg.action_dims
    )
    dataset_action_mean = dataset.stats["action"]["mean"].to(device)
    dataset_action_std = dataset.stats["action"]["std"].to(device)

    action_squared_error_sum = np.zeros(cfg.action_dims, dtype=np.float64)
    action_valid_count = np.zeros(cfg.action_dims, dtype=np.int64)
    ssim_sum = 0.0
    psnr_sum = 0.0
    image_count = 0
    saved_image_count = 0
    processed_batches = 0

    num_batches = len(dataloader)
    progress = tqdm(dataloader, total=num_batches, desc="Unified validation")
    for batch_index, batch in enumerate(progress):
        if batch_index >= num_batches:
            break
        batch = move_tensors_to_device(batch, device)

        normalized_gt_actions = batch["action"]
        raw_gt_actions = (
            normalized_gt_actions * (dataset_action_std + 1e-8)
            + dataset_action_mean
        )
        batch["action"] = raw_gt_actions
        # infer() moves predicted actions to CPU before denormalization.
        batch["action.mean"] = model_action_mean
        batch["action.std"] = model_action_std

        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            predicted_actions, predicted_images = policy.infer(
                batch,
                num_inference_steps=cfg.num_inference_steps,
            )
            # print(predicted_actions.shape, predicted_images.shape)

        predicted_actions = predicted_actions.float().numpy()
        raw_gt_actions = raw_gt_actions.detach().cpu().float().numpy()
        num_action_steps = min(
            predicted_actions.shape[1], raw_gt_actions.shape[1]
        )
        num_action_dims = min(
            cfg.action_dims,
            predicted_actions.shape[2],
            raw_gt_actions.shape[2],
        )
        valid_mask = get_valid_action_mask(
            batch,
            predicted_actions.shape[0],
            num_action_steps,
        ).numpy()
        squared_error = (
            predicted_actions[:, :num_action_steps, :num_action_dims]
            - raw_gt_actions[:, :num_action_steps, :num_action_dims]
        ) ** 2
        # print(predicted_actions[0, 0, :num_action_dims], raw_gt_actions[0, 0, :num_action_dims])
        action_squared_error_sum[:num_action_dims] += (
            squared_error * valid_mask[..., None]
        ).sum(axis=(0, 1), dtype=np.float64)
        valid_count = int(valid_mask.sum())
        action_valid_count[:num_action_dims] += valid_count

        predicted_rgb = prepare_rgb_images(predicted_images)
        target_rgb = prepare_rgb_images(
            batch["last_image"].detach().cpu().numpy()
        )
        if predicted_rgb.shape[0] != target_rgb.shape[0]:
            raise ValueError(
                f"Image batch sizes differ: {predicted_rgb.shape[0]} vs "
                f"{target_rgb.shape[0]}"
            )

        batch_ssim = []
        batch_psnr = []
        for sample_index, (predicted_image, target_image) in enumerate(
            zip(predicted_rgb, target_rgb, strict=True)
        ):
            ssim = compute_ssim(predicted_image, target_image)
            psnr = compute_psnr(predicted_image, target_image)
            batch_ssim.append(ssim)
            batch_psnr.append(psnr)
            ssim_sum += ssim
            psnr_sum += psnr
            image_count += 1
            if cfg.save_images and saved_image_count < cfg.max_saved_images:
                image_path = image_output_dir / (
                    f"batch_{batch_index:05d}_sample_{sample_index:02d}"
                    f"_ssim_{ssim:.4f}_psnr_{psnr:.2f}.png"
                )
                save_comparison(predicted_image, target_image, image_path)
                saved_image_count += 1

        processed_batches += 1
        current_action_count = action_valid_count[:num_action_dims]
        current_action_mse = np.divide(
            action_squared_error_sum[:num_action_dims],
            current_action_count,
            out=np.full(num_action_dims, np.nan, dtype=np.float64),
            where=current_action_count > 0,
        )
        progress.set_postfix(
            ssim=f"{np.mean(batch_ssim):.4f}",
            mean_ssim=f"{ssim_sum / image_count:.4f}",
            psnr_db=f"{np.mean(batch_psnr):.2f}",
            mean_psnr_db=f"{psnr_sum / image_count:.2f}",
            action_mse=f"{np.nanmean(current_action_mse):.6f}",
        )

    action_mse_per_dim = np.divide(
        action_squared_error_sum,
        action_valid_count,
        out=np.full(cfg.action_dims, np.nan, dtype=np.float64),
        where=action_valid_count > 0,
    )
    mean_action_mse = float(np.nanmean(action_mse_per_dim))
    mean_ssim = float(ssim_sum / image_count)
    mean_psnr = float(psnr_sum / image_count)

    metrics = {
        "checkpoint": str(checkpoint_path),
        "data_mix": cfg.data_mix,
        "validation_start_batch": cfg.validation_start_batch,
        "processed_batches": processed_batches,
        "image_count": image_count,
        "num_inference_steps": cfg.num_inference_steps,
        "action_dims": cfg.action_dims,
        "action_valid_count_per_dim": action_valid_count.tolist(),
        "action_mse_per_dim": action_mse_per_dim.tolist(),
        "mean_action_mse": mean_action_mse,
        "mean_image_ssim": mean_ssim,
        "mean_image_psnr_db": mean_psnr,
    }
    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, ensure_ascii=False, indent=2)

    print("\nValidation complete")
    print(f"Processed batches: {processed_batches}")
    print(f"Images: {image_count}")
    print(f"Action MSE per dimension: {action_mse_per_dim}")
    print(f"Mean action MSE: {mean_action_mse:.8f}")
    print(f"Mean image SSIM: {mean_ssim:.8f}")
    print(f"Mean image PSNR: {mean_psnr:.8f} dB")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    validate()
