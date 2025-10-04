import torch

weight_path = "/home/v-zuoleili/Pretrain/pi0/model.pt"
weights = torch.load(weight_path, map_location="cpu")
# print(weights.keys())
new_weights = {}
for k, v in weights.items():
    if "gemma_expert" in k:
        new_k = k.replace("model.paligemma_with_expert.gemma_expert.", "")
        new_weights[new_k] = v

print(new_weights.keys())
torch.save(new_weights, "/home/v-zuoleili/Pretrain/pi0/pi0_gemma_expert_only.pt")