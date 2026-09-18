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

import json
import logging
import warnings
from copy import deepcopy
from hashlib import sha256


_OPTIMIZER_GROUP_MISMATCHES = (
    "different number of parameter groups",
    "parameter group that doesn't match the size",
)
OPTIMIZER_GROUP_SIGNATURE_KEY = "optimizer_group_signature"
DATA_PARALLEL_WORLD_SIZE_KEY = "data_parallel_world_size"


def optimizer_group_signature(model, optimizer) -> str:
    """Hash ordered parameter names and shapes for every optimizer group."""
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    topology = []
    for index, group in enumerate(optimizer.param_groups):
        parameters = []
        for parameter in group["params"]:
            name = names_by_id.get(id(parameter))
            if name is None:
                raise ValueError(
                    "Optimizer contains a parameter that is not registered on the model."
                )
            parameters.append((name, list(parameter.shape)))
        topology.append(
            {
                "group_name": group.get("group_name", f"group_{index}"),
                "parameters": parameters,
            }
        )
    payload = json.dumps(topology, separators=(",", ":"), ensure_ascii=True)
    return f"sha256:{sha256(payload.encode('utf-8')).hexdigest()}"


def load_checkpoint_with_optimizer_fallback(
    model_engine,
    checkpoint_dir,
    expected_optimizer_signature: str,
    expected_data_parallel_world_size: int,
    lr_scheduler=None,
):
    """Resume optimizer state only when its saved parameter topology matches exactly."""
    fresh_scheduler_state = (
        deepcopy(lr_scheduler.state_dict()) if lr_scheduler is not None else None
    )
    load_path, client_state = model_engine.load_checkpoint(
        checkpoint_dir,
        load_optimizer_states=False,
        load_lr_scheduler_states=False,
        load_module_strict=False,
    )
    if load_path is None or client_state is None:
        return load_path, client_state, False

    saved_signature = client_state.get(OPTIMIZER_GROUP_SIGNATURE_KEY)
    saved_world_size = client_state.get(DATA_PARALLEL_WORLD_SIZE_KEY)
    incompatibilities = []
    if saved_signature is None:
        incompatibilities.append("the checkpoint predates optimizer topology signatures")
    elif saved_signature != expected_optimizer_signature:
        incompatibilities.append("the saved optimizer parameter topology differs")
    if saved_world_size is None:
        incompatibilities.append("the checkpoint does not record its data-parallel world size")
    elif saved_world_size != expected_data_parallel_world_size:
        incompatibilities.append(
            f"data-parallel world size changed from {saved_world_size} "
            f"to {expected_data_parallel_world_size}"
        )
    if incompatibilities:
        logging.warning(
            "Restored model weights only because %s. Optimizer moments will be rebuilt.",
            "; ".join(incompatibilities),
        )
        return load_path, client_state, False

    try:
        load_path, client_state = model_engine.load_checkpoint(
            checkpoint_dir,
            load_optimizer_states=True,
            load_lr_scheduler_states=True,
            load_module_strict=False,
        )
        return load_path, client_state, True
    except ValueError as exc:
        if not any(message in str(exc) for message in _OPTIMIZER_GROUP_MISMATCHES):
            raise
        logging.warning(
            "Checkpoint optimizer groups do not match the current model. Restoring model "
            "weights only and rebuilding optimizer/scheduler state: %s",
            exc,
        )
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(fresh_scheduler_state)
        return load_path, client_state, False


def align_fresh_scheduler_to_step(
    lr_scheduler,
    step: int,
    gradient_accumulation_steps: int,
) -> None:
    """Position a fresh scheduler after a model-only checkpoint restore."""
    if lr_scheduler is None:
        return
    optimizer_step = step // max(1, gradient_accumulation_steps)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Detected call of `lr_scheduler.step\\(\\)` before `optimizer.step\\(\\)`",
        )
        warnings.filterwarnings(
            "ignore",
            message="The epoch parameter in `scheduler.step\\(\\)` was not necessary",
        )
        lr_scheduler.step(optimizer_step)
    logging.warning(
        "Fresh optimizer state is using the LR schedule at optimizer step %d.",
        optimizer_step,
    )
