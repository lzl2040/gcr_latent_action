import torch
from diffusers import SanaPipeline, SanaVideoPipeline, DPMSolverMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)


model_id = "/Data/lzl/huggingface/SANA-Video_2B_480p_diffusers"
pipe = SanaVideoPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=8.0)
pipe.vae.to(torch.float32)
pipe.text_encoder.to(torch.bfloat16)
pipe.to("cuda")
model_score = 30

prompt = "Evening, backlight, side lighting, soft light, high contrast, mid-shot, centered composition, clean solo shot, warm color. A young Caucasian man stands in a forest, golden light glimmers on his hair as sunlight filters through the leaves. He wears a light shirt, wind gently blowing his hair and collar, light dances across his face with his movements. The background is blurred, with dappled light and soft tree shadows in the distance. The camera focuses on his lifted gaze, clear and emotional."
negative_prompt = "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
motion_prompt = f" motion score: {model_score}."
prompt = prompt + motion_prompt

video = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    frames=81,
    guidance_scale=6,
    num_inference_steps=10,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "sana_video.mp4", fps=16)
