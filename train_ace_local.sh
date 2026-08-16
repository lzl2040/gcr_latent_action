# Perception <-> physical contrastive pre-training (robo_contrast).
# conda activate lerobot_v2 && bash train_ace_local.sh
#
# GPU note: pick free devices. `nvidia-smi` first -- this is a shared machine.
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=256
export TOKENIZERS_PARALLELISM=false
NUM_GPUS=${NUM_GPUS:-4}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3} deepspeed --num_gpus=${NUM_GPUS} --master_port=29601 lerobot/scripts/dps_train_contrast.py \
    --deepspeed="./ds_zero2_contrast.json" \
    --policy.type="robo_contrast" \
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
    --dataset.dataset_size_one_epoch=1000000 \
    --num_workers=12 \
    --output_dir="robo_contrast" \
    --steps=60_0000 \
    --log_freq=20 \
    --policy.chunk_size=16 \
    --policy.frame_horizon=16 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=25000 \
    --policy.scheduler_platform_steps=20000 \
    --policy.optimizer_lr=1e-4 \
    --wandb.project="robo_contrast" \
    --job_name="perception_physical_contrast" \
    --weight_resume=false \
    --task_type="train_contrastive"
    # --wandb.enable=true \
