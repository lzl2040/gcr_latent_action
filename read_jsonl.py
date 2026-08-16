import json
import numpy as np
def read_jsonl(file_path):
    """Reads a JSONL file and returns a list of dictionaries.
    
    Args:
        file_path: Path to the JSONL file.
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            stats = json.loads(line)["stats"]
            std = np.array(stats["action"]["std"])
            if np.isnan(std).any() or np.isinf(std).any():
                print(f"Warning: Found NaN or Inf in action std for episode {stats}")
            data.append(json.loads(line))
    return data

path = "/Data/lerobot_data_ort6d/bridge_orig_lerobot/meta/episodes_stats.jsonl"
episodes_stats = read_jsonl(path)
print(episodes_stats[0])