import torch.nn as nn
import torch
from dataclasses import dataclass, asdict, replace
from typing import Optional, Dict, Any
import torch
import random

@dataclass
class BaseCondition:

    def to_dict(self, skip_underscore: bool = True) -> Dict[str, Any]:
        data = asdict(self)
        if skip_underscore:
            data = {k: v for k, v in data.items() if not k.startswith("_")}
        return data


# ------------------------------------------------
# Text2WorldCondition
# ------------------------------------------------

@dataclass
class Text2WorldCondition(BaseCondition):

    crossattn_emb: Optional[torch.Tensor] = None
    data_type: str = "video"
    padding_mask: Optional[torch.Tensor] = None
    fps: Optional[torch.Tensor] = None

    def edit_data_type(self, data_type: str):
        """修改 data_type 并返回新对象"""
        return replace(self, data_type=data_type)

    @property
    def is_video(self):
        return self.data_type == "video"


@dataclass
class Video2WorldCondition(Text2WorldCondition):

    use_video_condition: bool = False

    gt_frames: Optional[torch.Tensor] = None
    condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None

    def set_video_condition(
        self,
        gt_frames: torch.Tensor,
        random_min_num_conditional_frames: int = 0,
        random_max_num_conditional_frames: int = 4,
        num_conditional_frames: Optional[int] = None,
        conditional_frames_probs: Optional[Dict[int, float]] = None,
    ):
        """
        生成 video conditioning mask
        gt_frames shape: [B,C,T,H,W]
        """

        B, C, T, H, W = gt_frames.shape

        mask = torch.zeros(
            B, 1, T, H, W,
            dtype=gt_frames.dtype,
            device=gt_frames.device
        )

        # image batch
        if T == 1:
            num_conditional_frames_B = torch.zeros(B, dtype=torch.int32)

        else:

            if num_conditional_frames is not None:

                num_conditional_frames_B = torch.ones(B, dtype=torch.int32) * num_conditional_frames

            elif conditional_frames_probs is not None:

                frames_options = list(conditional_frames_probs.keys())
                weights = list(conditional_frames_probs.values())

                num_conditional_frames_B = torch.tensor(
                    random.choices(frames_options, weights=weights, k=B),
                    dtype=torch.int32
                )

            else:

                num_conditional_frames_B = torch.randint(
                    random_min_num_conditional_frames,
                    random_max_num_conditional_frames + 1,
                    (B,)
                )

        for i in range(B):
            mask[i, :, : num_conditional_frames_B[i], :, :] = 1

        return replace(
            self,
            gt_frames=gt_frames,
            condition_video_input_mask_B_C_T_H_W=mask
        )

    def edit_for_inference(
        self,
        is_cfg_conditional: bool = True,
        num_conditional_frames: int = 1
    ):
        cond = self.set_video_condition(
            gt_frames=self.gt_frames,
            num_conditional_frames=num_conditional_frames
        )

        if not is_cfg_conditional:
            cond.use_video_condition = True

        return cond

class Video2WorldConditioner(nn.Module):

    def __init__(self, text_dropout=0.2, flag_dropout=0.2):
        super().__init__()

        self.text_dropout = text_dropout
        self.flag_dropout = flag_dropout

    def forward(self, batch):

        output = {}

        # -----------------------
        # fps  (ReMapkey)
        # -----------------------
        fps = batch["fps"]
        output["fps"] = fps


        # -----------------------
        # padding_mask (ReMapkey)
        # -----------------------
        padding_mask = batch["padding_mask"]
        output["padding_mask"] = padding_mask


        # -----------------------
        # text (TextAttr)
        # -----------------------
        text_emb = batch["t5_text_embeddings"]

        B = text_emb.shape[0]

        keep_mask = torch.bernoulli(
            (1 - self.text_dropout) * torch.ones(B, device=text_emb.device)
        )

        keep_mask = keep_mask.view(B, *[1]*(text_emb.dim()-1))

        text_emb = keep_mask * text_emb

        output["crossattn_emb"] = text_emb


        # -----------------------
        # use_video_condition (BooleanFlag)
        # -----------------------
        flag = torch.bernoulli(
            (1 - self.flag_dropout) * torch.ones(1, device=fps.device)
        ).bool()

        output["use_video_condition"] = flag


        return Video2WorldCondition(**output)