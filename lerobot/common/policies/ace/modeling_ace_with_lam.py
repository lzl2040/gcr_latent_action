from torch import Tensor, nn
import torch

class ACE_Latent_Model(nn.Module):
    def __init__(self, lam_model, ace_model):
        super().__init__()
        self.lam = lam_model.vlm
        self.ace_model = ace_model
        self.sc_token_idx = lam_model.sc_token_idx
        self.action_token_idx = lam_model.action_token_idx
        
        # need not to update parameter
        self.lam.requires_grad_(False)
    
    def generate_token_mask(self, input_ids):
        sc_token_ids = torch.tensor(self.sc_token_idx, device=input_ids.device)
        act_token_ids = torch.tensor(self.action_token_idx, device=input_ids.device)
        act_token_mask = torch.isin(input_ids, act_token_ids)
        sc_token_mask = torch.isin(input_ids, sc_token_ids)
        return sc_token_mask, act_token_mask
    
    def extract_latent_embeddings(self, batch: dict[str, Tensor], device):
        pixel_values = batch["pixel_values"].to(device=device)
        input_ids = batch["input_ids"].to(device=device) # 对于224分辨率图像，每个image占64个token
        attention_mask = batch["attention_mask"].to(device=device, dtype=input_ids.dtype)
        output = self.lam(
            input_ids=input_ids,
            # labels=labels,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        # output_hidden_states = output
        output_hidden_states = output.hidden_states # num_layers + 1
        # logits = output.logits
        sc_token_mask, act_token_mask = self.generate_token_mask(input_ids)
        # get token embeddings
        # torch.Size([128, 1024]) torch.Size([4, 1024])
        last_hidden_states = output_hidden_states[-1]
        del output_hidden_states
        # last_hidden_states = self.vlm.model.language_model.norm(last_hidden_states)
        
        sc_embeddings = last_hidden_states[sc_token_mask]
        act_embeddings = last_hidden_states[act_token_mask]
        # print(act_embeddings.shape, torch.sum(act_token_mask), self.action_token_idx, torch.unique(input_ids))
        # sc_logits = logits[sc_token_mask]
        # act_logits = logits[act_token_mask]
        # print(sc_logits.shape, act_logits.shape, logits.shape, )
        hidden_size = sc_embeddings.shape[-1]
        bsize = input_ids.shape[0]
        sc_embeddings = sc_embeddings.view(bsize, -1, hidden_size).to(dtype=input_ids.dtype)
        act_embeddings = act_embeddings.view(bsize, -1, hidden_size).to(dtype=input_ids.dtype)
        latent_embeddings = torch.cat([sc_embeddings, act_embeddings], dim=1)
        return latent_embeddings
    
    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch, device):
        with torch.no_grad():
            latent_action_embeddings = self.extract_latent_embeddings(batch, device)
        
        