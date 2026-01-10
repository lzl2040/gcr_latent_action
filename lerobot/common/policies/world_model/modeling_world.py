import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModel, InternVLForConditionalGeneration, InternVLProcessor

from lerobot.common.constants import ACTION, OBS_ROBOT

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.world_model.configuration_world_model import LatentWorldModelConfig
from lerobot.common.policies.world_model.video_world_model import VideoWorldModel

def pad_vector(vector, new_dim):
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

class LatentWorldModel(PreTrainedPolicy):
    config_class = LatentWorldModelConfig
    name = "latent_wm"
    def __init__(
        self,
        config: LatentWorldModelConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.future_latent_encoder = InternVLForConditionalGeneration.from_pretrained(self.config.vlm_path,
                                                                    # config=vlm_config,
                                                                    local_files_only=True,
                                                                    trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.vlm_path,
                                                                    # config=vlm_config,
                                                                    local_files_only=True,
                                                                    trust_remote_code=True)
        
        self.world_decoder_model = VideoWorldModel(config)
        
        new_tokens = self.config.new_tokens
        self.tokenizer.add_tokens(self.config.new_tokens)
        self.cp_sc_token_idx = [self.tokenizer(new_tokens[0], add_special_tokens=False).input_ids[0]]
        self.cp_act_token_idx =  [self.tokenizer(new_tokens[1], add_special_tokens=False).input_ids[0]]
        
        # reduce gpu memory
        self.future_latent_encoder.model.language_model._set_gradient_checkpointing()
        self.future_latent_encoder.model.vision_tower.gradient_checkpointing = True
        self.future_latent_encoder.model.vision_tower.encoder.gradient_checkpointing = True
        # add it: 31G -> 24G
        if self.config.freeze_vision_encoder:
            print(f"Freeze VLM image encoder")
            self.future_latent_encoder.model.vision_tower.requires_grad_(False)
        self.sc_token_idx = config.sc_token_idx
        self.action_token_idx = config.action_token_idx
        self.img_token_id = self.future_latent_encoder.config.image_token_id
        print(f"In model: CP_IMG token idx: {self.cp_sc_token_idx}, CP_ACT token idx: {self.cp_act_token_idx}")
        print(f"In dataset: CP_IMG token idx: {self.config.sc_token_idx}, CP_ACT token idx: {self.config.action_token_idx}")

        self.dtype = torch.bfloat16
    
    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()
    
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        print("not use")
    

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    def convert_to_dtype(self, vector:torch.Tensor):
        if not isinstance(vector, type(None)):
            if vector.is_floating_point():
                vector = vector.to(dtype=self.dtype)
        return vector
    
    def generate_token_mask(self, input_ids):
        sc_token_ids = torch.tensor(self.sc_token_idx, device=input_ids.device)
        act_token_ids = torch.tensor(self.action_token_idx, device=input_ids.device)
        img_token_ids = torch.tensor(self.img_token_id, device=input_ids.device)
        act_token_mask = torch.isin(input_ids, act_token_ids)
        sc_token_mask = torch.isin(input_ids, sc_token_ids)
        img_token_mask = torch.isin(input_ids, img_token_ids)
        return sc_token_mask, act_token_mask, img_token_mask

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"] # 对于224分辨率图像，每个image占64个token
        attention_mask = batch["attention_mask"]
        future_imgs = batch["video_tensor"]
        actions = self.prepare_action(batch)
        actions = self.convert_to_dtype(actions)
        action_is_pad = batch.get("action_is_pad")
        
        sc_token_mask, act_token_mask, img_token_mask = self.generate_token_mask(input_ids)
        # print(pixel_values.shape, input_ids.shape, attention_mask.shape) # bs
        output = self.future_latent_encoder(
            input_ids=input_ids,
            # labels=labels,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        img_embeds = output.hidden_states[0][img_token_mask]  # N_img_token x D
        sc_embeds = output.hidden_states[-1][sc_token_mask]
        act_embeds = output.hidden_states[-1][act_token_mask]
        hidden_size = sc_embeds.shape[-1]
        bsize = input_ids.shape[0]
        sc_embeds = sc_embeds.view(bsize, -1, hidden_size)
        act_embeds = act_embeds.view(bsize, -1, hidden_size)
        img_embeds = img_embeds.view(bsize, -1, hidden_size)
        # print(img_embeds.shape)
        # prompt_embeds = torch.cat([img_embeds, sc_embeds, act_embeds], dim=1)
        future_imgs = future_imgs.permute(0, 2, 1, 3, 4)
        # print(future_imgs.shape)
        losses = self.world_decoder_model(img_embeds, sc_embeds, act_embeds, future_imgs, actions)
        # print(sc_embeds.shape, act_embeds.shape, img_embeds.shape)
        # print(img_embeds.shape) # 640 2048
        # print(losses["action_loss"].shape, batch["action_mask"].shape)
        # action_mask = batch["action_mask"].unsqueeze(1)
        # losses["action_loss"] = losses["action_loss"] * action_mask
        losses["action_loss"] = losses["action_loss"].mean()
        # print(action_mask.shape, losses["action_loss"].shape)
        loss = losses["action_loss"] + self.config.img_loss_weight * losses["video_loss"]
        loss_dict = {}
        loss_dict["total_loss"] = loss.item()
        loss_dict["action_loss"] = losses["action_loss"].item()
        loss_dict["video_loss"] = losses["video_loss"].item()
        loss_dict["language_loss"] = 0
        return loss, loss_dict
        
        
        