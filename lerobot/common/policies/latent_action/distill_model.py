import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, AutoModelForVision2Seq
# Qwen3VLForConditionalGeneration

class DistillModel(nn.Module):
    def __init__(self, teacher_model, student_model):
        super().__init__()
        self.teacher_model = teacher_model
        self.student_model = student_model
        # self.subtask_teacher_model = Qwen3VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
        # )
        # self.subtask_teacher_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        #     "Qwen/Qwen2.5-VL-3B-Instruct", dtype="auto", device_map="auto"
        # )
        # self.subtask_teacher_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        # self.subtask_teacher_model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        # self.subtask_teacher_processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
        # self.proj = nn.Linear(2048, 2048 // 4)  # 共享的映射层
        # print("Freeze Teacher model")
        for param in self.student_model.parameters():
            param.data = param.data.bfloat16()
        self.teacher_model.eval()
        self.teacher_model.requires_grad_(False)
        # self.subtask_teacher_model.requires_grad_(False)

    def generate_sub_task(self, images, tasks):
        sub_tasks = []
        n = len(tasks)
        for i in range(n):
            task = tasks[i]
            image = images[i]
            sub_tasks.append(f"Subtask:{task}")
            # print(images.shape)

            # prompt = f"You are an expert robot task planner. \n Given a high-level instruction: {task}, and the current visual observation, determine what the robot should do next — describe the **next subtask** in one short sentence. \nFormat: Subtask: <description>"
            # messages = [
            #     {
            #         "role": "user",
            #         "content": [
            #             {
            #                 "type": "image",
            #                 "image": image,
            #             },
            #             {"type": "text", "text": prompt},
            #         ],
            #     }
            # ]
            # # Preparation for inference
            # inputs = self.subtask_teacher_processor.apply_chat_template(
            #     messages,
            #     tokenize=True,
            #     add_generation_prompt=True,
            #     return_dict=True,
            #     return_tensors="pt"
            # ).to(self.subtask_teacher_model.device)

            # # Inference: Generation of the output
            # generated_ids = self.subtask_teacher_model.generate(**inputs, max_new_tokens=128)
            # generated_ids_trimmed = [
            #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            # ]
            # output_text = self.subtask_teacher_processor.batch_decode(
            #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            # )
            # # print(output_text)
            # sub_tasks.append(output_text[0])
        return sub_tasks

    def kl_mse_loss(
        self,
        student_h: torch.Tensor,
        teacher_h: torch.Tensor,
        student_logits:torch.Tensor,
        teacher_logits:torch.Tensor,
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
        log_p = F.log_softmax(student_logits, dim=-1)
        q = F.softmax(teacher_logits, dim=-1)

        kl = kl_loss_fn(log_p, q)

        # ---------- 合并 ----------
        loss = mse + alpha * kl
        # loss = mse
        # loss = kl
        # return loss, {"mse": mse.item(), "kl": kl.item()}
        return loss, {"mse": mse.item(), "kl": 0.0}
        # return loss, {"mse": 0.0, "kl": kl.item()}

    def get_optim_params(self) -> dict:
        return self.parameters()

    def forward(self, batch):
        with torch.no_grad():
            teacher_latent_emebedings = self.teacher_model.extract_latent_embeddings(batch)
        teacher_latent_emebedings = teacher_latent_emebedings.detach()
        
        sub_tasks = self.generate_sub_task(batch["observation.images.primary"], batch["task"])
        batch["sub_tasks"] = sub_tasks
        student_latent_embeddings, lg_loss = self.student_model.extract_vlm_hidden_states(batch)
        # 检查
        # print(student_latent_embeddings.requires_grad)  # True
        # print(teacher_latent_emebedings.requires_grad)  # False
        # print(torch.isnan(student_latent_embeddings).any(), torch.isnan(teacher_latent_emebedings).any())
        # final norm: 110, 165
        # wo norm: 10, 70
        # print(student_latent_embeddings.abs().max(), teacher_latent_emebedings.abs().max())
        # student_logits = self.proj(student_latent_embeddings)
        # teacher_logits = self.proj(teacher_latent_emebedings)
        student_logits = student_latent_embeddings
        teacher_logits = teacher_latent_emebedings
        student_logits = student_logits / (student_logits.norm(dim=-1, keepdim=True) + 1e-6)
        teacher_logits = teacher_logits / (teacher_logits.norm(dim=-1, keepdim=True) + 1e-6)
        # student_logits = student_latent_embeddings / (student_latent_embeddings.norm(dim=-1, keepdim=True) + 1e-6)
        # teacher_logits = teacher_latent_emebedings / (teacher_latent_emebedings.norm(dim=-1, keepdim=True) + 1e-6)

        loss, loss_dict = self.kl_mse_loss(student_h=student_latent_embeddings, 
                                           teacher_h=teacher_latent_emebedings,
                                           student_logits=student_logits,
                                           teacher_logits=teacher_logits,
                                           alpha=1,
                                           temperature=3)
        # loss = loss + 0.5 * lg_loss
        lg_loss = torch.tensor(0.0, device=loss.device)
        loss_dict["lg_loss"] = lg_loss
        return loss, loss_dict
