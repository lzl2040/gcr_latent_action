# Perception <-> physical contrastive pre-training (robo_contrast).
# conda activate lerobot_v2 && bash train_ace_local.sh
#
# GPU note: this is a shared machine, so run `nvidia-smi` and select free devices with
#   CUDA_VISIBLE_DEVICES=0,3 bash train_ace_local.sh
# Do NOT reintroduce `--num_gpus`/`--include`: deepspeed *ignores* CUDA_VISIBLE_DEVICES when
# either is passed and rewrites it to 0..N-1, which silently lands the job on whichever GPUs
# happen to be numbered first -- including ones another user is already filling.
export LEROBOT_VIDEO_DECODER_CACHE_SIZE=256
export TOKENIZERS_PARALLELISM=false
# Other jobs come and go on these GPUs, so keep our own footprint defragmented.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# Default to whichever GPUs are actually free, since some of them usually belong to someone
# else. Override explicitly with CUDA_VISIBLE_DEVICES=... when you want specific devices.
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', ' '$2 < 5000 {printf "%s%s", sep, $1; sep=","}')
fi
if [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    echo "No GPU has less than 5 GB in use; refusing to start. Check nvidia-smi." >&2
    exit 1
fi
export CUDA_VISIBLE_DEVICES
echo "devices=${CUDA_VISIBLE_DEVICES}"
# The machine is shared, and a run that is killed can leave dataloader workers holding the
# rendezvous socket, so a fixed port eventually fails with EADDRINUSE. Pick a free one.
MASTER_PORT=${MASTER_PORT:-$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")}
echo "master_port=${MASTER_PORT}"

deepspeed --master_port=${MASTER_PORT} lerobot/scripts/dps_train_contrast.py \
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
    --eval_freq=250 \
    --policy.chunk_size=16 \
    --policy.frame_horizon=16 \
    --policy.scheduler_warmup_steps=500 \
    --policy.scheduler_decay_steps=25000 \
    --policy.scheduler_platform_steps=20000 \
    --policy.optimizer_lr=1e-4 \
    --wandb.project="robo_contrast" \
    --job_name="perception_physical_contrast" \
    --weight_resume=false \
    --task_type="train_contrastive" \
    "$@"
    # --wandb.enable=true \
