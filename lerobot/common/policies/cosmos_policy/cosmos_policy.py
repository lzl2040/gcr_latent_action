from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.cosmos_policy.configuration_cosmos_policy import CosmosPolicyConfig
from lerobot.common.policies.cosmos_policy.wan2pt1 import Wan2pt1VAEInterface
from lerobot.common.policies.cosmos_policy.conditioner import Video2WorldConditioner
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, Qwen2_5_VLForConditionalGeneration
# from transformers.utils.generic import check_model_inputs
from transformers.utils import TransformersKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from torch import Tensor, nn
import torch
from typing import Any, Callable, Optional, Union
from collections import deque

NUM_EMBEDDING_PADDING_TOKENS = 512    
IS_PREPROCESSED_KEY = "is_preprocessed"
NUM_CONDITIONAL_FRAMES_KEY: str = "num_conditional_frames"


def replace_latent_with_action_chunk(
    x0: torch.Tensor, action_chunk: torch.Tensor, action_indices: torch.Tensor
) -> torch.Tensor:
    """
    Replaces the image latent (at the specified action index) in clean input image latents x0 with the action chunk.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and action_chunk has shape (B=32, chunk_size=8, action_dim=7).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,action_indices,:,:] with the action chunk,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        action_chunk (torch.Tensor): Ground-truth action chunk.
        action_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    action_image_latent = x0[batch_indices, :, action_indices, :, :]

    # Create a new tensor with the same shape as action_image_latent, filled with zeros
    result = torch.zeros_like(action_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = action_image_latent.shape

    # Flatten action_chunk (preserving batch dimension)
    flat_action = action_chunk.reshape(batch_size, -1)
    num_action_elements = flat_action.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the action elements
    assert num_action_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full action chunk: {num_action_elements} action elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the action tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_action_elements - 1) // num_action_elements

    # Repeat the action tensor along dimension 1
    repeated_action = flat_action.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_action = repeated_action[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the action chunk values into the beginning of the flattened result
    flat_result[:, :] = repeated_action

    # Reshape back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, action_indices, :, :] = result

    return new_x0


def replace_latent_with_proprio(x0: torch.Tensor, proprio: torch.Tensor, proprio_indices: torch.Tensor) -> torch.Tensor:
    """
    Replaces the image latent (at the specified proprio index) in clean input image latents x0 with the proprio.

    Example:
    Let's say x0 has shape (B=32, C'=16, T', H'=28, W'=28) and proprio has shape (B=32, proprio_dim=9).
    Then, this function will overwrite the (C'=16, H'=28, W'=28) volume at x0[:,:,proprio_indices,:,:] with the proprio,
    repeating it as many times as needed to fill the entire volume.

    Args:
        x0 (torch.Tensor): Clean image latents.
        proprio (torch.Tensor): Ground-truth proprio.
        proprio_indices (torch.Tensor): Batch indices of the image latents to replace.

    Returns:
        torch.Tensor: Modified image latents.
    """
    # Get latent to be replaced
    batch_indices = torch.arange(x0.shape[0], device=x0.device)
    proprio_image_latent = x0[batch_indices, :, proprio_indices, :, :]

    # Create a new tensor with the same shape as proprio_image_latent, filled with zeros
    result = torch.zeros_like(proprio_image_latent)

    # Get shapes
    batch_size, latent_channels, latent_h, latent_w = proprio_image_latent.shape

    # Get number of proprio elements
    # print(proprio[0].shape)
    num_proprio_elements = proprio.shape[1]

    # Calculate total elements in the target tensor (per batch)
    latent_elements = latent_channels * latent_h * latent_w

    # Check that there is enough room in the target tensor for all the proprio elements
    assert num_proprio_elements <= latent_elements, (
        f"Not enough room in the latent tensor for the full proprio: {num_proprio_elements} proprio elements > {latent_elements} latent elements!"
    )

    # Calculate how many times we need to repeat the proprio tensor
    # The expression below is a concise way of doing ceiling division to get the correct number of repeats
    num_repeats = (latent_elements + num_proprio_elements - 1) // num_proprio_elements

    # Repeat the proprio tensor along dimension 1
    repeated_proprio = proprio.repeat(1, num_repeats)

    # Take only what we need to fill the result tensor
    repeated_proprio = repeated_proprio[:, :latent_elements]

    # Reshape the target tensor to put all channel and spatial dimensions together
    flat_result = result.reshape(batch_size, -1)

    # Place the proprio values into the beginning of the flattened result
    flat_result[:, :] = repeated_proprio

    # Reshape latent back to original shape
    result = flat_result.reshape(batch_size, latent_channels, latent_h, latent_w)

    # Get final latents tensor
    new_x0 = x0
    new_x0[batch_indices, :, proprio_indices, :, :] = result

    return new_x0


class CosmosPolicyModel(PreTrainedPolicy):
    config_class = CosmosPolicyConfig
    name = "cosmos_policy"
    def __init__(
        self,
        config: CosmosPolicyConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        super().__init__(config)
        self.config = config
        config.validate_features()
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(config.text_encoder_path, 
                                                                            local_files_only=True,
                                                                            trust_remote_code=True
        )
        self.text_encoder.lm_head = nn.Identity()
        # self.text_encoder.forward = qwen3_forward.__get__(self.text_encoder)
        self.text_processor = AutoProcessor.from_pretrained(config.text_encoder_path)
        self.vae = Wan2pt1VAEInterface(config)
        self.conditioner = Video2WorldConditioner(text_dropout=0.2, flag_dropout=0.2)
        
        self.pad_id = self.text_processor.tokenizer.pad_token_id
        self.input_data_key = "video"
        self.dtype = torch.bfloat16
        
    
    def reset(self):
        """This should be called whenever the environment is reset."""
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()
    
    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        print("not use")
    
    def mean_normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Mean normalize a tensor by subtracting the mean and dividing by the standard deviation.

        Args:
        tensor (torch.tensor): The tensor to normalize

        Returns:
        torch.tensor: The normalized tensor
        """
        return (tensor - tensor.mean(dim=-1, keepdim=True)) / (tensor.std(dim=-1, keepdim=True) + 1e-8)
    
    def compute_text_embeddings(self, tasks, device):
        input_ids_batch = []
        # print(len(tasks))
        for sample_idx in range(len(tasks)):
            conversations = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a helpful assistant who will provide prompts to an image generator.",
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": tasks[sample_idx],
                        }
                    ],
                },
            ]
            tokenizer_output = self.text_processor.apply_chat_template(
                conversations,
                tokenize=True,
                return_dict=True,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            input_ids = tokenizer_output["input_ids"][0] # list
            # Do padding or truncation
            if NUM_EMBEDDING_PADDING_TOKENS > len(input_ids):
                # Do padding:
                pad_len = NUM_EMBEDDING_PADDING_TOKENS - len(input_ids)
                input_ids = input_ids + [self.pad_id] * pad_len
            else:
                # Do truncation:
                input_ids = input_ids[:NUM_EMBEDDING_PADDING_TOKENS]
            input_ids = torch.LongTensor(input_ids).to(device="cuda")
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch, dim=0).to(device)
        
        # Compute text embeddings
        # self.model = self.model.to(self.device)
        with torch.no_grad():
            outputs_batch = self.text_encoder(input_ids_batch, output_hidden_states=True)
        hidden_states = outputs_batch["hidden_states"]
        # print(hidden_states[0].shape) # batch length hidden_size, torch.Size([4, 512, 2560])

        # # Skip the embeddings of the system prompt
        # hidden_states = hidden_states[:, num_system_prompt_tokens:]

        # Now compute the normalized embeddings
        normalized_hidden_states = []
        # print(len(hidden_states))
        for layer_idx in range(1, len(hidden_states)):
            normalized_state = self.mean_normalize(hidden_states[layer_idx])
            normalized_hidden_states.append(normalized_state)

        # mean pooling: default to cosmos policy
        text_embeddings = torch.stack(normalized_hidden_states)
        text_embeddings = text_embeddings.mean(dim=0)
        return text_embeddings    
    
    def preprocess_video(self, data_batch, input_key: str = None):
        input_key = self.input_data_key if input_key is None else input_key
        # only handle video batch
        if input_key in data_batch:
            # Check if the data has already been normalized and avoid re-normalizing
            if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
                assert torch.is_floating_point(data_batch[input_key]), "Video data is not in float format."
                assert torch.all((data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)), (
                    f"Video data is not in the range [-1, 1]. get data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
                )
            else:
                assert data_batch[input_key].dtype == torch.uint8, "Video data is not in uint8 format."
                data_batch[input_key] = data_batch[input_key] / 127.5 - 1.0
                data_batch[IS_PREPROCESSED_KEY] = True

            # 4 * (self.config.state_t - 1) + 1
            # expected_length = self.text_processor.tokenizer.get_pixel_num_frames(self.config.state_t)
            # original_length = data_batch[input_key].shape[2]
            # assert original_length == expected_length, (
            #     f"Input video length doesn't match expected length specified by state_t: {original_length} != {expected_length}"
            # )
            
    def get_data_and_condition(self, data_batch, device):
        self.preprocess_video(data_batch)
        raw_state = data_batch[self.input_data_key].to(dtype=self.dtype) # only consider video. the code from cosmos policy also consider image
        latent_state = self.vae.encode(raw_state).to(device=device, dtype=self.dtype) # bs latent_dim t_dim h_dim w_dim, torch.Size([4, 16, 10, 32, 32])
        condition = self.conditioner(data_batch)
        condition = condition.edit_data_type("video")
        condition = condition.set_video_condition(
            gt_frames=latent_state,
            random_min_num_conditional_frames=self.config.min_num_conditional_frames,
            random_max_num_conditional_frames=self.config.max_num_conditional_frames,
            num_conditional_frames=data_batch.get(NUM_CONDITIONAL_FRAMES_KEY, None),
            conditional_frames_probs=self.config.conditional_frames_probs,
        )
        # print(condition.condition_video_input_mask_B_C_T_H_W.shape) # 4 1 10 32 32
        
        # fill state and action value into image
        if "proprio" in data_batch and torch.all(
            data_batch["current_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["proprio"],
                proprio_indices=data_batch["current_proprio_latent_idx"],
            )
        if "future_proprio" in data_batch and torch.all(
            data_batch["future_proprio_latent_idx"] != -1
        ):  # -1 indicates proprio is not used
            condition.gt_frames = replace_latent_with_proprio(
                condition.gt_frames,
                data_batch["future_proprio"],
                proprio_indices=data_batch["future_proprio_latent_idx"],
            )
        return raw_state, latent_state, condition
    
    def forward(self, data_batch : dict[str, Tensor]):
        images = data_batch["video"]
        device = images.device
        # compute text embeddings
        text_embeddings = self.compute_text_embeddings(data_batch["task"], device) # bs 512 hidden_size
        data_batch["t5_text_embeddings"] = text_embeddings.to(dtype=self.dtype)
        data_batch["t5_text_mask"] = torch.ones(text_embeddings.shape[0], text_embeddings.shape[1], device=device)
        raw_state, latent_state, condition = self.get_data_and_condition(data_batch, device)
        print(raw_state.shape)
        
        
        
        
        