import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer, AutoModel, InternVLForConditionalGeneration

from lerobot.common.constants import ACTION, OBS_ROBOT

from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype
from lerobot.common.policies.latent_action.configuration_latent_action import LatentActionConfig
from lerobot.common.policies.latent_action.action_decoder import PaliGemmaWithExpertConfig, ActionDecoderModel
from lerobot.common.policies.latent_action.image_decoder import ImagePredictionModel


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

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

class LatentActionModel(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = LatentActionConfig
    name = "latent_act"

    def __init__(
        self,
        config: LatentActionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.vlm = InternVLForConditionalGeneration.from_pretrained(self.config.vlm_path,
                                                                    local_files_only=True,
                                                                    trust_remote_code=True)
        # gradient_checkpointing: add it, bs=1, max gpu=44G
        # wo it, bs=1, max_gpu=64G
        self.vlm.model.language_model._set_gradient_checkpointing()
        self.vlm.model.vision_tower.gradient_checkpointing = True
        self.vlm.model.vision_tower.encoder.gradient_checkpointing = True

        self.sc_token_idx = config.sc_token_idx
        self.action_token_idx = config.action_token_idx
        self.uni_decoder = UniDecoder(config)

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
        # next-token prediction, so skip the first token
        sc_token_mask = torch.isin(input_ids[:, 1:], sc_token_ids)
        act_token_mask = torch.isin(input_ids[:, 1:], act_token_ids)
        bs = sc_token_mask.shape[0]
        pad = torch.zeros(bs, 1, dtype=torch.bool, device=act_token_ids.device)
        # print(act_token_mask.shape, pad.shape)
        act_token_mask = torch.cat([act_token_mask, pad], dim=1)
        sc_token_mask = torch.cat([sc_token_mask, pad], dim=1)
        # print(sc_token_mask.sum().item(), act_token_mask.sum().item())
        return sc_token_mask, act_token_mask
    
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        # print(batch["video_len"])
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        video_len = batch["video_len"]
        first_image = batch["first_image"]
        last_image = batch["last_image"]
        # print(first_image.shape, torch.max(first_image)) # 0-255
        actions = self.prepare_action(batch)
        actions = self.convert_to_dtype(actions)
        bsize = input_ids.shape[0]
        # torch.Size([B*T, 3, 224, 224]) torch.Size([1, 3266]) torch.Size([1, 3266])
        # print(pixel_values.shape, input_ids.shape, attention_mask.shape)
        output = self.vlm.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        output_hidden_states = output.hidden_states
        sc_token_mask, act_token_mask = self.generate_token_mask(input_ids)
        # get token embeddings
        # torch.Size([128, 1024]) torch.Size([4, 1024])
        sc_embeddings = output_hidden_states[-1][sc_token_mask]
        act_embeddings = output_hidden_states[-1][act_token_mask]
        hidden_size = sc_embeddings.shape[-1]
        sc_embeddings = sc_embeddings.view(bsize, -1, hidden_size)
        act_embeddings = act_embeddings.view(bsize, -1, hidden_size)
        # print(sc_embeddings.shape, act_embeddings.shape)
        # feed them into the decoder
        # for image, feed sc_embedding and first image into the decoder
        # for action, feed sc_embeddings, act_embeddings into the decoder
        # pixel_values = pixel_values.view(bsize, -1, 3, h, w)
        loss_dict = {}
        actions_is_pad = batch.get("actions_id_pad")
        losses = self.uni_decoder(first_image,
                                  last_image, 
                                  sc_embeddings, 
                                  act_embeddings, 
                                  actions)
        action_loss = losses["action_loss"]
        image_loss = losses["action_loss"]
        loss_dict["action_losses_after_forward"] = action_loss.clone()

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            action_loss = action_loss * in_episode_bound.unsqueeze(-1)
            loss_dict["action_losses_after_in_ep_bound"] = action_loss.clone()

        # Remove padding
        action_loss = action_loss[:, :, : self.config.max_action_dim]
        loss_dict["action_losses_after_rm_padding"] = action_loss.clone()

        # For backward pass
        loss = action_loss.mean() + image_loss.mean()
        # For logging
        loss_dict["total_loss"] = loss.item()
        loss_dict["action_loss"] = action_loss.mean().item()
        loss_dict["image_loss"] = image_loss.mean().item()

        return loss, loss_dict

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

class UniDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.dtype = torch.bfloat16

        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = ActionDecoderModel(paligemma_with_export_config, config.action_expert_path)

        # Projections are float32
        self.con_proj = nn.Linear(self.config.vlm_token_dim, self.config.img_dim)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)

        # image decoder
        self.image_decoder = ImagePredictionModel(config)

        self.dtype = torch.bfloat16
    
    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=self.dtype,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=self.dtype, device=device)

    def embed_prefix_for_action(self, images, con_embeddings):
        embs = []
        pad_masks = []
        att_masks = []
        bsize = con_embeddings.shape[0]
        # for img in images:
        #     img_emb = self.paligemma_with_expert.embed_image(img)
        #     img_emb = img_emb.to(dtype=torch.bfloat16)
        #     # Normalize image embeddings
        #     img_emb_dim = img_emb.shape[-1]
        #     img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

        #     bsize, num_img_embs = img_emb.shape[:2]
        #     embs.append(img_emb)
        #     # Create attention masks so that image tokens attend to each other
        #     att_masks += [0] * num_img_embs

        #     img_mask = torch.ones(bsize, dtype=torch.bool, device=img_emb.device)
        #     img_mask = img_mask[:, None].expand(bsize, num_img_embs)
        #     pad_masks.append(img_mask)

        # torch.Size([2, 256, 2048]) torch.Size([2, 66, 1024])
        # print(embs[0].shape, con_embeddings.shape)
        con_embeddings = self.con_proj(con_embeddings)
        embs.append(con_embeddings)
        num_con = con_embeddings.shape[1]
        att_masks += [0] * num_con
        con_mask = torch.ones(bsize, num_con, dtype=torch.bool, device=con_embeddings.device)
        pad_masks.append(con_mask)
        # print(pad_masks[0].shape, pad_masks[-1].shape)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks
    
    def embed_suffix_for_action(self, noisy_actions, timestep):
        embs = []
        pad_masks = []
        att_masks = []
        dtype = noisy_actions.dtype
        device = noisy_actions.device

        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # print(f"pad mask shape is: {pad_masks.shape}")
        # print(f"att mask shape is: {att_masks.shape}")
        return embs, pad_masks, att_masks

    def forward(
        self, first_image, last_image, sc_embedding, act_embeddings, 
        actions, action_noise=None, action_time=None
    ) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if action_noise is None:
            action_noise = self.sample_noise(actions.shape, actions.device).to(dtype=self.dtype)

        if action_time is None:
            action_time = self.sample_time(actions.shape[0], actions.device).to(dtype=self.dtype)

        action_time_expanded = action_time[:, None, None]
        actions = actions.to(dtype=self.dtype)
        x_t = action_time_expanded * action_noise + (1 - action_time_expanded) * actions
        u_t = action_noise - actions
        con_embeddings = torch.cat([sc_embedding, act_embeddings], dim = 1)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix_for_action(
            first_image, con_embeddings=con_embeddings
        )
        # torch.Size([2, 578, 2048])
        # print(prefix_embs.shape)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix_for_action(x_t, action_time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        # print(suffix_out.shape)
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # Original openpi code, upcast attention output
        # suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = suffix_out.to(dtype=self.dtype)
        v_t = self.action_out_proj(suffix_out)

        losses = {}
        losses["action_loss"] = F.mse_loss(u_t, v_t, reduction="none")
        # image predict
        losses["image_loss"] = self.image_decoder(sc_embedding, first_image, last_image)
        return losses
