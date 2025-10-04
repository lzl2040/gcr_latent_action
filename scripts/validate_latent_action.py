from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.common.policies.factory import make_policy
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.vt_dataset import MultiDatasetforDistTraining, extra_collate_fn
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.datasets.utils import cycle

import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import logging
import time
import os
import glob
import json
import functools
from pathlib import Path
from datetime import datetime
from pprint import pformat
from termcolor import colored

import numpy as np
from scipy.spatial.transform import Rotation as R
import copy
from tqdm import tqdm
import cv2

def act_delta(actions, batch):
    assert isinstance(actions, np.ndarray) , "actions must be a numpy array"
    # "observation.state"
    states = batch["observation.state"].cpu()
    state_mean = batch["state.mean"].cpu()
    state_std = batch["state.std"].cpu()
    states = states * (state_std + 1e-8) + state_mean
    states = states.cpu().numpy()
    # predict state
    p_states = []
    a_states = []
    batch_size = states.shape[0]
    # print(batch['action'].shape, actions.shape)
    # normed_action = copy.deepcopy( batch["action"].cpu().numpy())
    # batch["action"] = batch["action"].cpu().to(dtype=torch.float64)*(batch["action.std"].cpu().to(dtype=torch.float64) + 1e-8) + batch["action.mean"].cpu().to(dtype=torch.float64)
        
    for i in range(batch_size):
        state = states[i]
        p_state_i = []
        p_state = np.zeros(6)
        p_state[:3] = state[:3]
        x, y, z, w = state[3:7]
        r_mat = R.from_quat([x, y, z, w]).as_matrix()
        for j in range(actions.shape[1]):
            p_state[:3] += actions[i, j, :3]
            r_mat = r_mat @ R.from_euler('xyz', actions[i, j, 3:]).as_matrix()
            p_state[3:] = R.from_matrix(r_mat).as_euler('xyz', degrees=False)
            predict_state = copy.deepcopy(p_state)
            p_state_i.append(predict_state)
            del predict_state
            
        predict_state_i = copy.deepcopy(p_state_i)  
        p_states.append(predict_state_i)
        del predict_state_i
        a_state_i = []
        a_state = np.zeros(6)
        a_state[:3] = state[:3]
        x, y, z, w = state[3:7]
        r_mat = R.from_quat([x, y, z, w]).as_matrix()
        # actual_action = copy.deepcopy(batch['action']).cpu()[i]
        # action_std = batch["action.std"].cpu()
        # action_mean = batch["action.mean"].cpu()
        # normed_action = actual_action*(action_std + 1e-8) + action_mean
        # denormed_action = (normed_action-action_mean)/(action_std+1e-8)
        
        actual_act = batch['action'][i].cpu().numpy()
        for j in range(actual_act.shape[0]):
            a_state[:3] += actual_act[j, :3]
            r_mat = r_mat @ R.from_euler('xyz', actual_act[j, 3:6]).as_matrix()
            a_state[3:] = R.from_matrix(r_mat).as_euler('xyz', degrees=False)
            actual_state = copy.deepcopy(a_state)
            a_state_i.append(actual_state)
            del actual_state
        actual_state_i = copy.deepcopy(a_state_i)
        a_states.append(actual_state_i)
        del actual_state_i
    # print(f"predict state: {p_state_i[:5]}")
    # print(f"actual state: {a_state_i[:5]}")
    # print(f"predict action: {actions[-1][0][:6]}")
    # print(f"Ori action: {batch['ori_action'].cpu().numpy()[-1][0][:6]}")
    # print(f"Denormed action: {batch['action'].cpu().numpy()[-1][0][:6]}")
    # print(f"Normed action: {normed_action[-1][0][:6]}")
    
    # print(f"Diff after denorm: {denormed_action - actual_action}\nmean: {torch.mean(denormed_action - actual_action)}, std: {torch.std(denormed_action - actual_action)}")
    # batch x sequence_length x 6
    return np.array(p_states), np.array(a_states)
            
def get_predict_error(p_states, a_states):
    error = p_states - a_states
    init_error = error[:, 0, :]
    step_5_error = error[:, 4, :]
    step_10_error = error[:, 9, :]
    step_15_error = error[:, 14, :]
    step_20_error = error[:, 19, :]
    step_25_error = error[:, 24, :]
    # print(np.mean(init_error-step_15_error))
    return init_error, step_5_error, step_10_error, step_15_error, step_20_error, step_25_error

def get_predict_action(batch, model, device):
    with torch.no_grad():
        actions, predicted_image = model.infer(batch)
    
    predictd_actions = []
    for i in range(actions.shape[0]):
        predictd_action = [action[:6] for action in actions[i]]
        predictd_actions.append(predictd_action)

    predictd_actions = np.array(predictd_actions)
    
    return predictd_actions, predicted_image

def concat_with_gap(img1, img2, gap=10, direction="horizontal", color=255):
    """
    img1, img2: numpy 数组 (H, W, C)，dtype=uint8
    gap: 间隔像素数
    direction: "horizontal" 或 "vertical"
    color: 间隔区域颜色（0=黑，255=白）
    """
    if direction == "horizontal":
        h = max(img1.shape[0], img2.shape[0])
        # 创建 gap 区域
        gap_block = np.ones((h, gap, img1.shape[2]), dtype=img1.dtype) * color
        # print(img1.shape, gap_block.shape, img2.shape)
        result = np.concatenate((img1, gap_block, img2), axis=1)
    else:  # vertical
        w = max(img1.shape[1], img2.shape[1])
        gap_block = np.ones((gap, w, img1.shape[2]), dtype=img1.dtype) * color
        result = np.concatenate((img1, gap_block, img2), axis=0)
    return result

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    device = "cuda:0"
    cfg.validate()
    if cfg.seed is not None:
        set_seed(cfg.seed)


    seed = cfg.seed
    image_transforms = ImageTransforms(cfg.dataset.image_transforms)
    wrist_image_transforms = ImageTransforms(cfg.dataset.wrist_image_transforms)
    print(f"image transforms:{image_transforms}")
    print(f"wrist image transforms:{wrist_image_transforms}")
    dataset = MultiDatasetforDistTraining(
        cfg=cfg, 
        image_transforms=image_transforms,
        wrist_image_transforms=wrist_image_transforms,
        seed=seed,
        data_mix=cfg.data_mix,
        vla2root_json="vla2root.json",
        is_ft=cfg.is_ft
        # vla2root_json="vla2root_bak_single.json"
    )

    json_path = "/home/v-zuoleili/Pretrain/latent-action/oxe_magic_soup_plus_stats.json"
    with open(json_path, "r") as f:
        json_data = json.load(f)

    action_mean = json_data["action"]["mean"]
    action_std = json_data["action"]["std"]
    state_mean = json_data["observation.state"]["mean"]
    state_std = json_data["observation.state"]["std"]

    data_action_mean = dataset.stats["action"]["mean"].to(device)
    data_action_std = dataset.stats["action"]["std"].to(device)
    data_state_mean = dataset.stats["observation.state"]["mean"].to(device)
    data_state_std = dataset.stats["observation.state"]["std"].to(device)
    print("Action Meta: \n", action_mean, action_std)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=2,
        collate_fn=extra_collate_fn,
        pin_memory=False,
    )

    cfg.policy.set_token_idx(dataset.cp_act_token_idx, dataset.cp_sc_token_idx)
    policy = make_policy(
        cfg=cfg.policy,
        device="cpu",
        ds_meta=dataset.meta,
        weight_pt_path=cfg.policy.pretrained_path
    )

    policy = policy.to(device)

    for params in policy.parameters():
        params.data = params.data.bfloat16()


    init_errors = []
    step_5_errors = []
    step_10_errors = []
    step_15_errors = []
    step_20_errors = []
    step_25_errors = []

    loader_cycler = cycle(dataloader)

    for i in tqdm(range(200)):
        batch = next(loader_cycler)
        # print(batch["pixel_values"].shape)
        # print(batch['source'])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        batch["action.mean"] = torch.from_numpy(np.array(action_mean))
        batch["action.std"] = torch.from_numpy(np.array(action_std))
        batch["state.mean"] = data_state_mean
        batch["state.std"] = data_state_std
        batch["action"] = batch["action"] * (data_action_std + 1e-8) + data_action_mean

        predicted_actions, predicted_images = get_predict_action(batch, policy, device)
        predict_state, actual_state = act_delta(predicted_actions, batch)

        init_error, step_5_error, step_10_error, step_15_error, step_20_error, step_25_error = get_predict_error(predict_state, actual_state)
        print(f"Batch {i}, Init error:{init_error}")
        init_errors.append(init_error)
        step_5_errors.append(step_5_error)
        step_10_errors.append(step_10_error)
        step_15_errors.append(step_15_error)
        step_20_errors.append(step_20_error)
        step_25_errors.append(step_25_error)

        # save predicted image and gt image
        actual_image = batch["last_image"]
        actual_image = actual_image.permute(0, 2, 3, 1)
        actual_image = actual_image.detach().cpu().numpy()
        actual_image = cv2.cvtColor(actual_image[0], cv2.COLOR_BGR2RGB)
        predicted_image = predicted_images[0] * 255
        concat_img = concat_with_gap(predicted_image, actual_image, gap=20, direction="horizontal", color=255)
        os.makedirs("predict", exist_ok=True)
        cv2.imwrite(f"predict/batch_{i}.png", concat_img)

    
    init_errors = np.array(init_errors)
    step_5_errors = np.array(step_5_errors)
    step_10_errors = np.array(step_10_errors)
    step_15_errors = np.array(step_15_errors)
    step_20_errors = np.array(step_20_errors)
    step_25_errors = np.array(step_25_errors)
    
    shape_error = init_errors.shape # 10000 x 6
    init_errors = init_errors.reshape(-1, shape_error[-1])
    step_5_errors = step_5_errors.reshape(-1, shape_error[-1])
    step_10_errors = step_10_errors.reshape(-1, shape_error[-1])
    step_15_errors = step_15_errors.reshape(-1, shape_error[-1])
    step_20_errors = step_20_errors.reshape(-1, shape_error[-1])
    step_25_errors = step_25_errors.reshape(-1, shape_error[-1])
    print(f"Each dim mean: \nInit:{np.mean(np.abs(init_errors), axis=0)}\nStep5:{np.mean(np.abs(step_5_errors), axis=0)}\nStep10:{np.mean(np.abs(step_10_errors), axis=0)}\nStep15:{np.mean(np.abs(step_15_errors), axis=0)}\nStep20:{np.mean(np.abs(step_20_errors), axis=0)}\nStep25:{np.mean(np.abs(step_25_errors), axis=0)}")
    print(f"Each dim std: \nInit:{np.std(init_errors, axis=0)}\nStep5:{np.std(step_5_errors, axis=0)}\nStep10:{np.std(step_10_errors, axis=0)}\nStep15:{np.std(step_15_errors, axis=0)}\nStep20:{np.std(step_20_errors, axis=0)}\nStep25:{np.std(step_25_errors, axis=0)}")
    print(f"Each dim Max: \nInit:{np.max(np.abs(init_errors), axis=0)}\nStep5:{np.max(np.abs(step_5_errors), axis=0)}\nStep10:{np.max(np.abs(step_10_errors), axis=0)}\nStep15:{np.max(np.abs(step_15_errors), axis=0)}\nStep20:{np.max(np.abs(step_20_errors), axis=0)}\nStep25:{np.max(np.abs(step_25_errors), axis=0)}")


if __name__ == "__main__":
    train()
