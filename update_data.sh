DATA_MIX="simpler_bridge"
# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --data_mix)
            DATA_MIX="$2"
            shift 2
            ;;
    esac
done
python -m lerobot.common.datasets.v21.convert_dataset_v20_to_v21 \
    --data_mix $DATA_MIX