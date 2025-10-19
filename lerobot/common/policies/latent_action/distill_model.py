import torch.nn as nn
import torch
import torch.nn.functional as F

class DistillModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        # print("Freeze Teacher model")
        # for param in self.teacher_model.parameters():
        #     param.requires_grad = False
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)

    def kl_mse_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        alpha: float = 1.0,
        temperature: float = 1.0,
        reduction: str = "batchmean",
    ):
        """
        KL + MSE 组合损失（完全使用 PyTorch 内置函数）

        Args:
            student_h: [B, L, D] 学生模型 hidden states
            teacher_h: [B, L, D] 教师模型 hidden states
            mask: [B, L] 可选，padding mask（0/1 或 bool）
            alpha: KL loss 权重
            temperature: KL 温度系数
            reduction: "none" | "mean" | "batchmean"
            detach_target: 是否对教师 hidden detach
        Returns:
            loss: 标量总损失
            stats: dict 包含 mse 和 kl 数值
        """

        mse_loss_fn = nn.MSELoss(reduction="mean")
        kl_loss_fn = nn.KLDivLoss(reduction=reduction)


        mse = mse_loss_fn(student_h, teacher_h)

        # ---------- KL 部分 ----------
        log_p = F.log_softmax(student_h / temperature, dim=-1)
        q = F.softmax(teacher_h / temperature, dim=-1)

        kl = kl_loss_fn(log_p, q) * (temperature ** 2)

        # ---------- 合并 ----------
        # loss = mse + alpha * kl
        loss = mse
        # return loss, {"mse": mse.item(), "kl": kl.item()}
        return loss, {"mse": mse.item(), "kl": 0.0}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch):
        with torch.no_grad():
            teacher_latent_emebedings = self.teacher_model.extract_latent_embeddings(batch)
        teacher_latent_emebedings = teacher_latent_emebedings.detach()
        student_latent_embeddings = self.student_model.extract_vlm_hidden_states(batch)
        # 检查
        # print(student_latent_embeddings.requires_grad)  # True
        # print(teacher_latent_emebedings.requires_grad)  # False
        # print(torch.isnan(student_latent_embeddings).any(), torch.isnan(teacher_latent_emebedings).any())
        # print(student_latent_embeddings.abs().max(), teacher_latent_emebedings.abs().max())
        loss, loss_dict = self.kl_mse_loss(student_h=student_latent_embeddings, 
                                           teacher_h=teacher_latent_emebedings,
                                           alpha=0.4,
                                           temperature=3)
        return loss, loss_dict
