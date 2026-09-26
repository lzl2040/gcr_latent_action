#!/usr/bin/env python

from __future__ import annotations

import gc
import json
import logging
import os
import random
import threading
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.fsdp import (
    BackwardPrefetch,
    FullyShardedDataParallel,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import CustomPolicy

from lerobot.common.utils.deepspeed_checkpoint import optimizer_group_signature

_LATEST_CHECKPOINT = "latest_checkpoint"
_METADATA_FILE = "metadata.json"
_TRAINER_STATE_FILE = "trainer_state.pt"
_FP8_RECIPES = {"tensorwise", "rowwise", "rowwise_with_gw_hp"}
_FP8_SCOPES = {"generation", "generation_vlm"}
_T = TypeVar("_T")
_DEFAULT_DISTRIBUTED_TIMEOUT_MINUTES = 60
_DEFAULT_CHECKPOINT_HEARTBEAT_SECONDS = 60
_CHECKPOINT_VISIBILITY_TIMEOUT_SECONDS = 60
_LOGGER = logging.getLogger(__name__)
FSDP_CHECKPOINT_IO_COMPAT_VERSION = 1


def make_distributed_timeout(
    minutes: str | int = _DEFAULT_DISTRIBUTED_TIMEOUT_MINUTES,
) -> timedelta:
    try:
        parsed_minutes = int(minutes)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"DISTRIBUTED_TIMEOUT_MINUTES must be a positive integer, got {minutes!r}."
        ) from exc
    if parsed_minutes <= 0:
        raise ValueError(
            f"DISTRIBUTED_TIMEOUT_MINUTES must be a positive integer, got {minutes!r}."
        )
    return timedelta(minutes=parsed_minutes)


def create_checkpoint_process_group(timeout: timedelta) -> dist.ProcessGroup:
    """Create a CPU process group for checkpoint planning and status collectives."""
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "The default distributed process group must be initialized before "
            "creating the checkpoint process group."
        )
    if dist.get_backend() == dist.Backend.GLOO:
        return dist.group.WORLD
    return dist.new_group(backend=dist.Backend.GLOO, timeout=timeout)


def _environment_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, str(default).lower())
    if value not in {"true", "false"}:
        raise ValueError(f"{name} must be true or false, got {value!r}.")
    return value == "true"


def _environment_int(name: str, default: int, *, allow_zero: bool = False) -> int:
    value = os.environ.get(name, str(default))
    try:
        parsed_value = int(value)
    except ValueError as exc:
        qualifier = "a non-negative" if allow_zero else "a positive"
        raise ValueError(f"{name} must be {qualifier} integer, got {value!r}.") from exc
    if parsed_value < 0 or (parsed_value == 0 and not allow_zero):
        qualifier = "a non-negative" if allow_zero else "a positive"
        raise ValueError(f"{name} must be {qualifier} integer, got {value!r}.")
    return parsed_value


@dataclass(frozen=True)
class FSDPCheckpointIOConfig:
    sync_files: bool = False
    thread_count: int = 1
    heartbeat_seconds: int = _DEFAULT_CHECKPOINT_HEARTBEAT_SECONDS

    @classmethod
    def from_environment(cls) -> FSDPCheckpointIOConfig:
        config = cls(
            sync_files=_environment_bool("FSDP_CHECKPOINT_SYNC_FILES", False),
            thread_count=_environment_int("FSDP_CHECKPOINT_THREADS", 1),
            heartbeat_seconds=_environment_int(
                "FSDP_CHECKPOINT_HEARTBEAT_SECONDS",
                _DEFAULT_CHECKPOINT_HEARTBEAT_SECONDS,
                allow_zero=True,
            ),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.thread_count <= 0:
            raise ValueError("FSDP checkpoint thread_count must be positive.")
        if self.heartbeat_seconds < 0:
            raise ValueError(
                "FSDP checkpoint heartbeat_seconds must be non-negative."
            )


@dataclass
class FSDPTrainingConfig:
    fp8: bool = True
    fp8_recipe: str = "rowwise_with_gw_hp"
    fp8_scope: str = "generation_vlm"
    fp8_emulate: bool = False
    fp8_min_features: int = 128
    min_wrap_params: int = 1_000_000
    forward_prefetch: bool = False
    limit_all_gathers: bool = True
    replicate_frozen_params: bool = True

    def validate(self) -> None:
        if self.fp8_recipe not in _FP8_RECIPES:
            raise ValueError(
                f"Unsupported FP8 recipe {self.fp8_recipe!r}; choose from "
                f"{sorted(_FP8_RECIPES)}."
            )
        if self.fp8_scope not in _FP8_SCOPES:
            raise ValueError(
                f"Unsupported FP8 scope {self.fp8_scope!r}; choose from "
                f"{sorted(_FP8_SCOPES)}."
            )
        if self.fp8_min_features <= 0:
            raise ValueError("fp8_min_features must be positive.")
        if self.min_wrap_params <= 0:
            raise ValueError("min_wrap_params must be positive.")


@dataclass(frozen=True)
class FSDPWrapSummary:
    wrapped_module_names: tuple[str, ...]
    trainable_parameters: int
    replicated_frozen_parameters: int


@dataclass(frozen=True)
class FSDPResumeState:
    checkpoint_dir: Path
    micro_step: int
    update_step: int
    epoch: int
    batch_in_epoch: int


def _require_native_fp8_device(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError("Native FP8 training requires a CUDA device.")
    capability = torch.cuda.get_device_capability(device)
    if capability[0] < 9:
        name = torch.cuda.get_device_name(device)
        raise RuntimeError(
            "Native TorchAO FP8 training requires compute capability 9.0 or newer "
            f"(H100-class); got {name} with capability {capability[0]}.{capability[1]}. "
            "Set fsdp.fp8_emulate=true only for correctness smoke tests."
        )


def _is_fp8_linear(
    module: nn.Module,
    fqn: str,
    config: FSDPTrainingConfig,
) -> bool:
    if type(module) is not nn.Linear:
        return False
    if "lora_" in fqn:
        return False
    in_scope = fqn.startswith("generation.")
    if config.fp8_scope == "generation_vlm":
        in_scope = in_scope or (
            fqn.startswith("understanding.")
            and ".language_model.layers." in f".{fqn}."
        )
    if not in_scope:
        return False
    if min(module.in_features, module.out_features) < config.fp8_min_features:
        return False
    return module.in_features % 16 == 0 and module.out_features % 16 == 0


def convert_policy_to_fp8(
    policy: nn.Module,
    config: FSDPTrainingConfig,
    device: torch.device,
) -> tuple[str, ...]:
    """Replace eligible Linear layers while retaining BF16 master parameters."""
    config.validate()
    if not config.fp8:
        return ()
    if not config.fp8_emulate:
        _require_native_fp8_device(device)

    try:
        from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    except ImportError as exc:
        raise ImportError(
            "FP8 training requires torchao>=0.15,<0.16; run the Stage 2 launcher "
            "dependency probe for the active PyTorch stack."
        ) from exc

    converted_names = tuple(
        name
        for name, module in policy.named_modules()
        if _is_fp8_linear(module, name, config)
    )
    if not converted_names:
        raise ValueError(
            f"No eligible Linear modules matched FP8 scope {config.fp8_scope!r}."
        )

    float8_config = replace(
        Float8LinearConfig.from_recipe_name(config.fp8_recipe),
        emulate=config.fp8_emulate,
        enable_fsdp_float8_all_gather=False,
    )
    convert_to_float8_training(
        policy,
        module_filter_fn=lambda module, fqn: _is_fp8_linear(module, fqn, config),
        config=float8_config,
    )
    return converted_names


def find_fsdp_wrap_modules(
    model: nn.Module,
    min_wrap_params: int,
) -> tuple[tuple[str, nn.Module], ...]:
    """Choose repeated trainable blocks without wrapping every Linear separately."""
    named_modules = dict(model.named_modules())
    selected: dict[str, nn.Module] = {}
    for parent_name, parent in named_modules.items():
        if not isinstance(parent, nn.ModuleList):
            continue
        for index, child in enumerate(parent):
            trainable = sum(
                parameter.numel()
                for parameter in child.parameters()
                if parameter.requires_grad
            )
            if trainable < min_wrap_params:
                continue
            child_name = f"{parent_name}.{index}" if parent_name else str(index)
            selected[child_name] = child

    # Prefer the outer repeated block if a model contains nested ModuleLists.
    kept: list[tuple[str, nn.Module]] = []
    for name in sorted(selected, key=lambda item: (item.count("."), item)):
        if any(name.startswith(f"{parent_name}.") for parent_name, _ in kept):
            continue
        kept.append((name, selected[name]))
    return tuple(kept)


def wrap_policy_with_fsdp(
    policy: nn.Module,
    config: FSDPTrainingConfig,
    device: torch.device,
) -> tuple[FullyShardedDataParallel, FSDPWrapSummary]:
    config.validate()
    wrap_modules = find_fsdp_wrap_modules(policy, config.min_wrap_params)
    wrap_ids = {id(module) for _, module in wrap_modules}
    frozen_parameters = tuple(
        parameter for parameter in policy.parameters() if not parameter.requires_grad
    )
    trainable_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    replicated_frozen_parameters = (
        sum(parameter.numel() for parameter in frozen_parameters)
        if config.replicate_frozen_params
        else 0
    )

    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
        keep_low_precision_grads=True,
    )
    fsdp_model = FullyShardedDataParallel(
        policy,
        auto_wrap_policy=CustomPolicy(lambda module: id(module) in wrap_ids),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        sync_module_states=False,
        forward_prefetch=config.forward_prefetch,
        limit_all_gathers=config.limit_all_gathers,
        use_orig_params=True,
        ignored_states=frozen_parameters if config.replicate_frozen_params else None,
    )
    summary = FSDPWrapSummary(
        wrapped_module_names=tuple(name for name, _ in wrap_modules),
        trainable_parameters=trainable_parameters,
        replicated_frozen_parameters=replicated_frozen_parameters,
    )
    return fsdp_model, summary


def _checkpoint_options() -> StateDictOptions:
    return StateDictOptions(
        full_state_dict=False,
        cpu_offload=True,
        strict=True,
    )


def _canonical_fsdp_name(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "")


@torch.no_grad()
def _restore_replicated_frozen_parameters(
    model: FullyShardedDataParallel,
    model_state: dict[str, Any],
) -> None:
    """FSDP does not copy ignored parameters in set_model_state_dict."""
    frozen_parameters = {
        _canonical_fsdp_name(name): parameter
        for name, parameter in model.module.named_parameters()
        if not parameter.requires_grad
    }
    missing = sorted(set(frozen_parameters) - set(model_state))
    if missing:
        preview = ", ".join(missing[:5])
        raise KeyError(
            "Checkpoint is missing replicated frozen parameters "
            f"({len(missing)} total): {preview}."
        )
    for name, parameter in frozen_parameters.items():
        value = model_state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(
                f"Replicated frozen parameter {name!r} must load as a Tensor, got "
                f"{type(value).__name__}."
            )
        if value.shape != parameter.shape:
            raise ValueError(
                f"Replicated frozen parameter {name!r} changed shape from "
                f"{tuple(value.shape)} to {tuple(parameter.shape)}."
            )
        parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _run_checkpoint_phase(
    phase: str,
    operation: Callable[[], _T],
    device: torch.device,
    process_group: dist.ProcessGroup | None = None,
) -> _T:
    """Run rank-local I/O and make every rank fail before the next collective."""
    rank = dist.get_rank(process_group)
    start_time = time.monotonic()
    if rank == 0:
        _LOGGER.info("Distributed checkpoint phase started: %s", phase)

    result = None
    local_exception = None
    try:
        result = operation()
    except Exception as exc:  # noqa: BLE001
        local_exception = exc

    failure_count = torch.tensor(
        int(local_exception is not None),
        device=(
            torch.device("cpu")
            if process_group is not None
            and dist.get_backend(process_group) == dist.Backend.GLOO
            else device
        ),
        dtype=torch.int32,
    )
    dist.all_reduce(
        failure_count,
        op=dist.ReduceOp.SUM,
        group=process_group,
    )
    if failure_count.item() > 0:
        local_error = (
            None
            if local_exception is None
            else {
                "rank": rank,
                "type": type(local_exception).__name__,
                "message": str(local_exception)[:2_000],
            }
        )
        gathered_errors = [None] * dist.get_world_size(process_group)
        dist.all_gather_object(
            gathered_errors,
            local_error,
            group=process_group,
        )
        details = "; ".join(
            f"rank {error['rank']}: {error['type']}: {error['message']}"
            for error in gathered_errors
            if error is not None
        )
        message = f"Distributed checkpoint phase {phase!r} failed: {details}"
        if local_exception is not None:
            raise RuntimeError(message) from local_exception
        raise RuntimeError(message)
    if rank == 0:
        _LOGGER.info(
            "Distributed checkpoint phase completed: %s (%.1f s)",
            phase,
            time.monotonic() - start_time,
        )
    return result


def _iter_local_tensors(
    value: Any,
    seen: set[int],
) -> Iterator[torch.Tensor]:
    value_id = id(value)
    if value_id in seen:
        return
    seen.add(value_id)

    to_local = getattr(value, "to_local", None)
    if callable(to_local):
        local_value = to_local()
        if isinstance(local_value, torch.Tensor):
            yield local_value
        return
    local_shards = getattr(value, "local_shards", None)
    if callable(local_shards):
        for shard in local_shards():
            tensor = getattr(shard, "tensor", None)
            if isinstance(tensor, torch.Tensor):
                yield tensor
        return
    if isinstance(value, torch.Tensor):
        yield value
        return
    if isinstance(value, Mapping):
        for nested_value in value.values():
            yield from _iter_local_tensors(nested_value, seen)
        return
    if isinstance(value, (list, tuple)):
        for nested_value in value:
            yield from _iter_local_tensors(nested_value, seen)


def _local_model_state_statistics(model_state: Mapping[str, Any]) -> tuple[int, int]:
    tensors = tuple(_iter_local_tensors(model_state, set()))
    return len(tensors), sum(
        tensor.numel() * tensor.element_size() for tensor in tensors
    )


def _checkpoint_file_summary(checkpoint_dir: Path) -> tuple[int, int]:
    files = tuple(
        path
        for path in checkpoint_dir.glob("*.distcp")
        if path.is_file()
    )
    return len(files), sum(path.stat().st_size for path in files)


@contextmanager
def _checkpoint_heartbeat(
    phase: str,
    interval_seconds: int,
    checkpoint_dir: Path | None = None,
) -> Iterator[None]:
    if interval_seconds == 0 or dist.get_rank() != 0:
        yield
        return

    stopped = threading.Event()
    start_time = time.monotonic()

    def log_progress() -> None:
        while not stopped.wait(interval_seconds):
            elapsed_seconds = time.monotonic() - start_time
            if checkpoint_dir is None:
                _LOGGER.info(
                    "Distributed checkpoint phase still running: %s (%.1f s)",
                    phase,
                    elapsed_seconds,
                )
                continue
            try:
                file_count, total_bytes = _checkpoint_file_summary(checkpoint_dir)
            except OSError as exc:
                _LOGGER.warning(
                    "Could not inspect checkpoint progress for %s after %.1f s: %s",
                    phase,
                    elapsed_seconds,
                    exc,
                )
                continue
            _LOGGER.info(
                "Distributed checkpoint phase still running: %s "
                "(%.1f s, %d shard files, %.2f GiB visible)",
                phase,
                elapsed_seconds,
                file_count,
                total_bytes / 2**30,
            )

    heartbeat = threading.Thread(
        target=log_progress,
        name="fsdp-checkpoint-heartbeat",
        daemon=True,
    )
    heartbeat.start()
    try:
        yield
    finally:
        stopped.set()
        heartbeat.join(timeout=1)


def _make_checkpoint_writer(
    checkpoint_dir: Path,
    checkpoint_io: FSDPCheckpointIOConfig,
) -> dcp.FileSystemWriter:
    checkpoint_io.validate()
    return dcp.FileSystemWriter(
        checkpoint_dir,
        single_file_per_rank=True,
        sync_files=checkpoint_io.sync_files,
        thread_count=checkpoint_io.thread_count,
        overwrite=False,
    )


def _verify_model_checkpoint(checkpoint_dir: Path) -> tuple[int, int]:
    metadata_path = checkpoint_dir / ".metadata"
    deadline = time.monotonic() + _CHECKPOINT_VISIBILITY_TIMEOUT_SECONDS
    while not metadata_path.is_file():
        if time.monotonic() >= deadline:
            raise FileNotFoundError(
                f"DCP model metadata was not visible after "
                f"{_CHECKPOINT_VISIBILITY_TIMEOUT_SECONDS} seconds: {metadata_path}."
            )
        time.sleep(1)

    metadata = dcp.FileSystemReader(checkpoint_dir).read_metadata()
    if not metadata.state_dict_metadata:
        raise RuntimeError(f"DCP model metadata is empty: {metadata_path}.")
    if not metadata.storage_data:
        raise RuntimeError(f"DCP model storage metadata is empty: {metadata_path}.")

    expected_sizes: dict[str, int] = {}
    for storage_info in metadata.storage_data.values():
        relative_path = storage_info.relative_path
        expected_sizes[relative_path] = max(
            expected_sizes.get(relative_path, 0),
            storage_info.offset + storage_info.length,
        )

    while True:
        missing = []
        truncated = []
        actual_bytes = 0
        for relative_path, expected_size in expected_sizes.items():
            path = checkpoint_dir / relative_path
            if not path.is_file():
                missing.append(relative_path)
                continue
            actual_size = path.stat().st_size
            actual_bytes += actual_size
            if actual_size < expected_size:
                truncated.append(
                    f"{relative_path} ({actual_size} < {expected_size} bytes)"
                )
        if not missing and not truncated:
            return len(expected_sizes), actual_bytes
        if time.monotonic() >= deadline:
            details = []
            if missing:
                details.append(f"missing={missing}")
            if truncated:
                details.append(f"truncated={truncated}")
            raise RuntimeError(
                "DCP model shards were not completely visible before checkpoint "
                f"publication: {'; '.join(details)}."
            )
        time.sleep(1)


def _local_training_state(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    device: torch.device,
) -> dict:
    return {
        "optimizer": optimizer.state_dict(),
        "optimizer_signature": optimizer_group_signature(model, optimizer),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(device),
    }


def save_fsdp_checkpoint(
    *,
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: str | Path,
    fsdp_config: FSDPTrainingConfig,
    training_geometry: dict[str, Any],
    micro_step: int,
    update_step: int,
    epoch: int,
    batch_in_epoch: int,
    device: torch.device,
    checkpoint_process_group: dist.ProcessGroup | None = None,
    checkpoint_io: FSDPCheckpointIOConfig | None = None,
) -> Path:
    """Save a reshardable model and same-world-size rank-local AdamW8bit states."""
    checkpoint_io = checkpoint_io or FSDPCheckpointIOConfig.from_environment()
    checkpoint_io.validate()
    output_dir = Path(output_dir)
    checkpoint_name = f"checkpoint_{update_step:08d}"
    checkpoint_dir = output_dir / checkpoint_name
    temporary_dir = output_dir / f".{checkpoint_name}.tmp"
    model_checkpoint_dir = temporary_dir / "model"

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    def prepare_directory() -> None:
        if rank != 0:
            return
        if checkpoint_dir.exists() or temporary_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite an existing FSDP checkpoint at {checkpoint_dir} "
                f"or {temporary_dir}."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary_dir.mkdir()

    _run_checkpoint_phase(
        "prepare save directory",
        prepare_directory,
        device,
        checkpoint_process_group,
    )

    def materialize_model_state() -> dict[str, Any]:
        with _checkpoint_heartbeat(
            "materialize model state",
            checkpoint_io.heartbeat_seconds,
        ):
            return get_model_state_dict(model, options=_checkpoint_options())

    model_state = _run_checkpoint_phase(
        "materialize model state",
        materialize_model_state,
        device,
        checkpoint_process_group,
    )
    tensor_count, local_bytes = _local_model_state_statistics(model_state)
    _LOGGER.info(
        "Rank %d materialized %d local model tensors (%.2f GiB)",
        rank,
        tensor_count,
        local_bytes / 2**30,
    )

    def write_model_shards() -> None:
        writer = _make_checkpoint_writer(model_checkpoint_dir, checkpoint_io)
        with _checkpoint_heartbeat(
            "write model shards",
            checkpoint_io.heartbeat_seconds,
            model_checkpoint_dir,
        ):
            dcp.save(
                {"model": model_state},
                storage_writer=writer,
                process_group=checkpoint_process_group,
            )

    _run_checkpoint_phase(
        "write model shards",
        write_model_shards,
        device,
        checkpoint_process_group,
    )

    def verify_model_checkpoint() -> None:
        if rank != 0:
            return
        file_count, total_bytes = _verify_model_checkpoint(model_checkpoint_dir)
        _LOGGER.info(
            "Verified DCP model checkpoint: %d shard files, %.2f GiB",
            file_count,
            total_bytes / 2**30,
        )

    _run_checkpoint_phase(
        "verify model checkpoint",
        verify_model_checkpoint,
        device,
        checkpoint_process_group,
    )
    model_state.clear()
    gc.collect()

    _run_checkpoint_phase(
        "save rank-local optimizer state",
        lambda: torch.save(
            _local_training_state(optimizer, model, device),
            temporary_dir / f"optimizer_rank_{rank:05d}.pt",
        ),
        device,
        checkpoint_process_group,
    )

    def save_metadata() -> None:
        if rank != 0:
            return
        metadata = {
            "format_version": 1,
            "micro_step": micro_step,
            "update_step": update_step,
            "epoch": epoch,
            "batch_in_epoch": batch_in_epoch,
            "world_size": world_size,
            "fsdp_config": asdict(fsdp_config),
            "checkpoint_io": asdict(checkpoint_io),
            "training_geometry": training_geometry,
        }
        (temporary_dir / _METADATA_FILE).write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        torch.save(
            {"scheduler": scheduler.state_dict() if scheduler is not None else None},
            temporary_dir / _TRAINER_STATE_FILE,
        )

    _run_checkpoint_phase(
        "save checkpoint metadata",
        save_metadata,
        device,
        checkpoint_process_group,
    )

    def publish_checkpoint() -> None:
        if rank != 0:
            return
        temporary_dir.rename(checkpoint_dir)
        pointer_tmp = output_dir / f".{_LATEST_CHECKPOINT}.tmp"
        pointer_tmp.write_text(checkpoint_name, encoding="utf-8")
        pointer_tmp.replace(output_dir / _LATEST_CHECKPOINT)

    _run_checkpoint_phase(
        "publish checkpoint",
        publish_checkpoint,
        device,
        checkpoint_process_group,
    )
    return checkpoint_dir


def resolve_latest_fsdp_checkpoint(output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    pointer = output_dir / _LATEST_CHECKPOINT
    if not pointer.is_file():
        raise FileNotFoundError(
            f"`weight_resume=true` was requested, but {pointer} does not exist."
        )
    checkpoint_dir = output_dir / pointer.read_text(encoding="utf-8").strip()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"The latest FSDP checkpoint pointer resolves to missing directory "
            f"{checkpoint_dir}."
        )
    return checkpoint_dir


def load_fsdp_checkpoint(
    *,
    model: FullyShardedDataParallel,
    optimizer: torch.optim.Optimizer,
    scheduler,
    output_dir: str | Path,
    fsdp_config: FSDPTrainingConfig,
    training_geometry: dict[str, Any],
    device: torch.device,
    checkpoint_process_group: dist.ProcessGroup | None = None,
    checkpoint_io: FSDPCheckpointIOConfig | None = None,
) -> FSDPResumeState:
    checkpoint_io = checkpoint_io or FSDPCheckpointIOConfig.from_environment()
    checkpoint_io.validate()

    def load_metadata() -> tuple[Path, dict[str, Any]]:
        checkpoint_dir = resolve_latest_fsdp_checkpoint(output_dir)
        metadata_path = checkpoint_dir / _METADATA_FILE
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"FSDP checkpoint metadata is missing: {metadata_path}."
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return checkpoint_dir, metadata

    checkpoint_dir, metadata = _run_checkpoint_phase(
        "load checkpoint metadata",
        load_metadata,
        device,
        checkpoint_process_group,
    )
    world_size = dist.get_world_size()
    if metadata.get("world_size") != world_size:
        raise ValueError(
            "AdamW8bit optimizer states are rank-local and require the same world size: "
            f"checkpoint={metadata.get('world_size')}, current={world_size}."
        )
    saved_fsdp_config = metadata.get("fsdp_config")
    current_fsdp_config = asdict(fsdp_config)
    if saved_fsdp_config != current_fsdp_config:
        raise ValueError(
            "FSDP/FP8 settings changed across resume. Checkpoint settings are "
            f"{saved_fsdp_config}, current settings are {current_fsdp_config}."
        )
    saved_training_geometry = metadata.get("training_geometry")
    if saved_training_geometry != training_geometry:
        raise ValueError(
            "Batching or sampler geometry changed across resume. Checkpoint geometry is "
            f"{saved_training_geometry}, current geometry is {training_geometry}."
        )

    def materialize_model_state() -> dict[str, Any]:
        with _checkpoint_heartbeat(
            "materialize model load state",
            checkpoint_io.heartbeat_seconds,
        ):
            return {
                "model": get_model_state_dict(model, options=_checkpoint_options())
            }

    model_state = _run_checkpoint_phase(
        "materialize model load state",
        materialize_model_state,
        device,
        checkpoint_process_group,
    )

    def read_model_shards() -> None:
        with _checkpoint_heartbeat(
            "read model shards",
            checkpoint_io.heartbeat_seconds,
            checkpoint_dir / "model",
        ):
            dcp.load(
                model_state,
                checkpoint_id=checkpoint_dir / "model",
                process_group=checkpoint_process_group,
            )

    _run_checkpoint_phase(
        "read model shards",
        read_model_shards,
        device,
        checkpoint_process_group,
    )

    def apply_model_state() -> None:
        set_model_state_dict(
            model,
            model_state["model"],
            options=_checkpoint_options(),
        )
        if fsdp_config.replicate_frozen_params:
            _restore_replicated_frozen_parameters(model, model_state["model"])

    _run_checkpoint_phase(
        "apply model state",
        apply_model_state,
        device,
        checkpoint_process_group,
    )
    model_state.clear()
    gc.collect()

    rank = dist.get_rank()

    def load_optimizer() -> dict[str, Any]:
        local_state_path = checkpoint_dir / f"optimizer_rank_{rank:05d}.pt"
        if not local_state_path.is_file():
            raise FileNotFoundError(
                f"Rank-local AdamW8bit state is missing: {local_state_path}."
            )
        local_state = torch.load(
            local_state_path,
            map_location="cpu",
            weights_only=False,
        )
        expected_signature = optimizer_group_signature(model, optimizer)
        if local_state.get("optimizer_signature") != expected_signature:
            raise ValueError(
                "Optimizer parameter groups changed across resume; refusing to attach "
                "8-bit moments to different parameters."
            )
        optimizer.load_state_dict(local_state["optimizer"])
        return local_state

    local_state = _run_checkpoint_phase(
        "load rank-local optimizer state",
        load_optimizer,
        device,
        checkpoint_process_group,
    )

    def load_trainer_state() -> None:
        trainer_state_path = checkpoint_dir / _TRAINER_STATE_FILE
        if not trainer_state_path.is_file():
            raise FileNotFoundError(
                f"FSDP scheduler state is missing: {trainer_state_path}."
            )
        trainer_state = torch.load(
            trainer_state_path,
            map_location="cpu",
            weights_only=False,
        )
        saved_scheduler = trainer_state.get("scheduler")
        if scheduler is None and saved_scheduler is not None:
            raise ValueError(
                "Checkpoint has a scheduler state but the current run does not."
            )
        if scheduler is not None and saved_scheduler is None:
            raise ValueError(
                "Current run has a scheduler but the checkpoint does not."
            )
        if scheduler is not None:
            scheduler.load_state_dict(saved_scheduler)

    _run_checkpoint_phase(
        "load scheduler state",
        load_trainer_state,
        device,
        checkpoint_process_group,
    )

    def restore_rng_state() -> None:
        random.setstate(local_state["python_rng_state"])
        np.random.set_state(local_state["numpy_rng_state"])
        torch.set_rng_state(local_state["torch_rng_state"])
        torch.cuda.set_rng_state(local_state["cuda_rng_state"], device)

    _run_checkpoint_phase(
        "restore random state",
        restore_rng_state,
        device,
        checkpoint_process_group,
    )
    return FSDPResumeState(
        checkpoint_dir=checkpoint_dir,
        micro_step=int(metadata["micro_step"]),
        update_step=int(metadata["update_step"]),
        epoch=int(metadata["epoch"]),
        batch_in_epoch=int(metadata["batch_in_epoch"]),
    )
