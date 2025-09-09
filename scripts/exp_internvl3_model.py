import math
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
# path = "/home/v-zuoleili/Pretrain/InternVL3_5-1B-HF"
# device_map = split_model(path)
# model = AutoModel.from_pretrained(
#     path,
#     torch_dtype=torch.bfloat16,
#     load_in_8bit=False,
#     low_cpu_mem_usage=True,
#     use_flash_attn=True,
#     trust_remote_code=True,
#     device_map=device_map).eval()
# tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)

# # set the max number of tiles in `max_num`
# pixel_values = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# generation_config = dict(max_new_tokens=1024, do_sample=True)

# # pure-text conversation (纯文本对话)
# question = 'Hello, who are you?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Can you tell me a story?'
# response, history = model.chat(tokenizer, None, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # single-image single-round conversation (单图单轮对话)
# question = '<image>\nPlease describe the image shortly.'
# response = model.chat(tokenizer, pixel_values, question, generation_config)
# print(f'User: {question}\nAssistant: {response}')

# # single-image multi-round conversation (单图多轮对话)
# question = '<image>\nPlease describe the image in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Please write a poem according to the image.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, combined images (多图多轮对话，拼接图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# question = '<image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # multi-image multi-round conversation, separate images (多图多轮对话，独立图像)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]

# question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'What are the similarities and differences between these two images.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list,
#                                history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# # batch inference, single image per sample (单图批处理)
# pixel_values1 = load_image('./examples/image1.jpg', max_num=12).to(torch.bfloat16).cuda()
# pixel_values2 = load_image('./examples/image2.jpg', max_num=12).to(torch.bfloat16).cuda()
# num_patches_list = [pixel_values1.size(0), pixel_values2.size(0)]
# pixel_values = torch.cat((pixel_values1, pixel_values2), dim=0)

# questions = ['<image>\nDescribe the image in detail.'] * len(num_patches_list)
# responses = model.batch_chat(tokenizer, pixel_values,
#                              num_patches_list=num_patches_list,
#                              questions=questions,
#                              generation_config=generation_config)
# for question, response in zip(questions, responses):
#     print(f'User: {question}\nAssistant: {response}')

# video multi-round conversation (视频多轮对话)
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

# video_path = "/home/v-zuoleili/Pretrain/InternVL3-1B-Instruct/examples/red-panda.mp4"
# pixel_values, num_patches_list = load_video(video_path, num_segments=8, max_num=1)
# pixel_values = pixel_values.to(torch.bfloat16).cuda()
# video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
# question = video_prefix + 'What is the red panda doing?'
# # Frame1: <image>\nFrame2: <image>\n...\nFrame8: <image>\n{question}
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=None, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

# question = 'Describe this video in detail.'
# response, history = model.chat(tokenizer, pixel_values, question, generation_config,
#                                num_patches_list=num_patches_list, history=history, return_history=True)
# print(f'User: {question}\nAssistant: {response}')

from transformers import InternVLForConditionalGeneration, InternVLConfig, InternVLProcessor, Trainer
from transformers.models.auto import CONFIG_MAPPING

intervl_config = CONFIG_MAPPING["internvl"](
    transformers_version="4.51.1",
    torch_dtype="bfloat16",
    model_type="internvl_chat",
    architectures=["InternVLChatModel"],
    auto_map={
        "AutoConfig": "configuration_internvl_chat.InternVLChatConfig",
        "AutoModel": "modeling_internvl_chat.InternVLChatModel",
        "AutoModelForCausalLM": "modeling_internvl_chat.InternVLChatModel",
    },
    downsample_ratio=0.5,
    dynamic_image_size=True,
    force_image_size=448,
    max_dynamic_patch=12,
    min_dynamic_patch=1,
    ps_version="v2",
    select_layer=-1,
    template="internvl2_5",
    use_backbone_lora=0,
    use_llm_lora=0,
    use_thumbnail=True,

    # ---------------------- LLM 配置 (Qwen3) ----------------------
    llm_config={
        "_name_or_path": "/root/codespace/checkpoints/Qwen3-0.6B",
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "torch_dtype": "bfloat16",
        "vocab_size": 151936,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "max_position_embeddings": 40960,
        "max_window_layers": 28,
        "hidden_act": "silu",
        "initializer_range": 0.02,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "use_cache": False,
        "attn_implementation": "flash_attention_2",
    },

    # ---------------------- 视觉模型配置 (InternViT) ----------------------
    vision_config={
        "architectures": ["InternVisionModel"],
        "model_type": "intern_vit_6b",
        "torch_dtype": "bfloat16",
        "image_size": 448,
        "num_channels": 3,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "patch_size": 14,
        "hidden_act": "gelu",
        "layer_norm_eps": 1e-06,
        "initializer_range": 0.02,
        "initializer_factor": 1.0,
        "drop_path_rate": 0.0,
        "dropout": 0.0,
        "qkv_bias": True,
        "tie_word_embeddings": True,
        "use_flash_attn": True,
    }
)
# config = InternVLConfig.from_pretrained(path)
# model = InternVLForConditionalGeneration(config=intervl_config)
torch_device = "cuda"
path = "/home/v-zuoleili/Pretrain/InternVL3_5-1B-HF"
model = InternVLForConditionalGeneration.from_pretrained(path)
processor = InternVLProcessor.from_pretrained(path)
# print(intervl_config)

# from transformers import AutoProcessor, AutoModelForImageTextToText
# model = AutoModelForImageTextToText.from_pretrained(path, trust_remote_code=True)
# processor = AutoProcessor.from_pretrained(path)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "https://cdn.britannica.com/61/93061-050-99147DCE/Statue-of-Liberty-Island-New-York-Bay.jpg",
            },
            {
                "type": "image",
                "url": "https://thumbs.dreamstime.com/b/golden-gate-bridge-san-francisco-purple-flowers-california-echium-candicans-36805947.jpg",
            },
            {"type": "text", "text": "These images depict two different landmarks. Can you identify them?"},
        ],
    },
]

inputs = processor.apply_chat_template(messages, 
                                       add_generation_prompt=True, tokenize=False, 
                                       return_dict=True, return_tensors="pt",
                                       image_size = 256)
                                       
# <|im_start|>user
# <IMG_CONTEXT>
# <IMG_CONTEXT>
# These images depict two different landmarks. Can you identify them?<|im_end|>
# <|im_start|>assistant

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# model_checkpoint = path
# quantization_config = BitsAndBytesConfig(load_in_4bit=True)
# processor = AutoProcessor.from_pretrained(model_checkpoint)
# model = AutoModelForImageTextToText.from_pretrained(model_checkpoint, quantization_config=quantization_config)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "url": "https://huggingface.co/datasets/hf-internal-testing/fixtures_videos/resolve/main/tennis.mp4",
            },
            {"type": "text", "text": "What type of shot is the man performing?"},
        ],
    }
]
# processor.video_processor["size"] = {
#     "height" : 256,
#     "width" : 256
# }
# must be 448
image_size = 224
processor.video_processor.size["height"] = image_size
processor.video_processor.size["width"] = image_size
processor.image_seq_length = 256 // 4
# model.config.vision_config.image_size = [image_size, image_size]
# model.config.vision_config.hidden_size = model.config.vision_config.hidden_size // 4
print(model.config)
# print(processor.video_processor)
inputs = processor.apply_chat_template(
    messages,
    return_tensors="pt",
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    num_frames=8,
).to(model.device, dtype=torch.float16)
print(inputs["pixel_values"].shape, inputs["input_ids"].shape, inputs["attention_mask"].shape)
output = model.generate(**inputs, max_new_tokens=25)
