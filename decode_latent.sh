# conda activate lerobot_v2_1
python scripts/vis_decode_frames.py \
    --policy.type="latent_act" \
    --policy.max_frame=16 \
    --policy.chunk_size=15 \
    --policy.n_action_steps=15 \
    --is_ft=true \
    --save_freq=2000 \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=true \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --dataset.parent_dir="/Data/lerobot_data/simulated" \
    --data_mix="simpler_bridge" \
    --dataset.sample_ratio=5 \
    --output_dir="qwen_flow" \
    --batch_size=1 \
    --steps=60_0000 \
    --policy.scheduler_warmup_steps=10000 \
    --policy.scheduler_decay_steps=25000 \
    --policy.scheduler_platform_steps=20000 \
    --policy.optimizer_lr=1e-4 \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.train_from_scratch=true \
    --wandb.project="fsdp_qwen_pi0_ft" \
    --job_name="debug_simpler_bridge" \
    --policy.pretrained_path="/Data/lzl/latent_action/0124_pretrain_latent_unfied_decoder/step80000.pt"
    # --wandb.enable=true \
    