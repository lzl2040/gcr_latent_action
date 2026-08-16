#!/bin/bash
conda run -n lerobot_v2 python -c "
from lerobot.common.datasets_v30.streaming_dataset import benchmark_lola_streaming_dataset
benchmark_lola_streaming_dataset()
"
