# distill_latent_action
# fsdp_train_uni_token
# conda activate lerobot_v2
# torchrun --nnodes=1 \
#     --nproc_per_node=4 \
#     --master_port=9912 \
# ms_buy_v30, 
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=256
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 --master_port=29601 lerobot/scripts/dps_train_ace.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="robo_clip" \
    --is_ft=false \
    --save_freq=2000 \
    --dataset.repo_id="whatever" \
    --dataset.image_transforms.enable=false \
    --dataset.wrist_image_transforms.enable=false \
    --dataset.wrist_image_transforms.is_primary=false \
    --dataset.processor="/Data/lzl/huggingface/InternVL3_5-2B-HF" \
    --dataset.parent_dir_v21="/Data/lerobot_data_ort6d" \
    --dataset.parent_dir_v30="/Data/lerobot_data_ort6d/v30" \
    --dataset.video_backend="torchcodec" \
    --data_mix="debug_research_data" \
    --dataset.sample_ratio=5 \
    --dataset.dataset_size_one_epoch=100000 \
    --output_dir="qwen_flow" \
    --steps=60_0000 \
    --log_freq=20 \
    --policy.scheduler_warmup_steps=10 \
    --policy.scheduler_decay_steps=25000 \
    --policy.scheduler_platform_steps=20000 \
    --policy.optimizer_lr=1e-4 \
    --wandb.project="fsdp_qwen_pi0_ft" \
    --job_name="debug_simpler_bridge" \
    --weight_resume=false \
    --policy.frozen_ace=false \
    --task_type="train_ace"
    # --wandb.enable=true \
    # --policy.pretrained_path="/Data/lzl/ace_weights/step_27k/mp_rank_00_model_states.pt" \
    