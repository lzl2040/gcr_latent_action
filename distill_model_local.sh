# distill_latent_action
# dps_distill_latent_action
# fsdp_train_uni_token
PRETRAINED_PATH=""
# torchrun --nnodes=1 \
#     --nproc_per_node=4 \
#     --master_port=9912 \
deepspeed \
    lerobot/scripts/dps_distill_latent_action.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="latent_act" \
    --policy2.type="pi05" \
    --policy.max_frame=16 \
    --is_ft=false \
    --save_freq=2000 \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=true \
    --dataset.wrist_image_transforms.enable=true \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --dataset.parent_dir="/Data/lerobot_data" \
    --data_mix="toy" \
    --dataset.sample_ratio=5 \
    --output_dir="qwen_flow" \
    --batch_size=5 \
    --steps=60_0000 \
    --log_freq=20 \
    --policy.scheduler_warmup_steps=10 \
    --policy.scheduler_decay_steps=25000 \
    --policy.scheduler_platform_steps=20000 \
    --policy.optimizer_lr=1e-4 \
    --policy.train_main_layers=0 \
    --policy.freeze_vision_encoder=true \
    --policy.train_expert_only=false \
    --policy.train_from_scratch=true \
    --policy.is_distill=true \
    --wandb.project="fsdp_qwen_pi0_ft" \
    --job_name="debug_simpler_bridge" \
    --policy.pretrained_path=$PRETRAINED_PATH \
    --policy2.pretrained_path="/Data/lzl/openpi/pytorch/pi05_base/model_new.pt" \
    # --wandb.enable=true \
    