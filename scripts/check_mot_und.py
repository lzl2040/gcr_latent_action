"""Numerically verify the custom UND branch against official Phi-4-Multimodal.

The check covers both official adapter modes:

* language: no vision LoRA;
* vision: SigLIP + image projector + rank-256 vision LoRA.

The production MoT needs a custom language stack to export every layer's pre-RoPE K/V. This
script proves that the custom path preserves the pretrained model instead of merely matching
its tensor shapes.

Run:
    CUDA_VISIBLE_DEVICES=2 python -u scripts/check_mot_und.py
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerobot.common.policies.mot.modeling_mot import MoTConfig  # noqa: E402
from lerobot.common.policies.mot.world_model import (  # noqa: E402
    MoTWorldModel,
    Phi4MMVisionEmbedding,
    WorldModelConfig,
)

PHI_DIR = "/Data/lzl/huggingface/Phi-4-multimodal-instruct"
IMAGE_TOKEN_ID = 200010


def _load_reference(phi_dir: str, device: str, dtype: torch.dtype):
    """Load the checkpoint's original remote-code model despite newer PEFT API changes."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("modeling_phi4mm.Phi4MMForCausalLM", phi_dir)
    # Current PEFT expects this generation hook on the wrapped bare model. It is never called
    # by this forward-only check, but older Phi remote code predates that PEFT requirement.
    bare_cls = cls.__init__.__globals__["Phi4MMModel"]
    if not hasattr(bare_cls, "prepare_inputs_for_generation"):
        bare_cls.prepare_inputs_for_generation = lambda self, *args, **kwargs: kwargs

    config = AutoConfig.from_pretrained(phi_dir, trust_remote_code=True)
    config._attn_implementation = "sdpa"
    config._attn_implementation_internal = "sdpa"
    model = cls.from_pretrained(
        phi_dir,
        config=config,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    return model.to(device).eval()


def reference_outputs(
    phi_dir: str,
    text_ids: torch.Tensor,
    pixels: torch.Tensor,
    device: str,
    dtype: torch.dtype,
):
    model = _load_reference(phi_dir, device, dtype)
    text_ids = text_ids.to(device)
    pixels = pixels.to(device=device, dtype=dtype)

    model.unset_lora_adapter()
    with torch.no_grad():
        language = model.model(
            input_ids=text_ids,
            use_cache=False,
            output_hidden_states=False,
        ).last_hidden_state

        # A square image produces one global crop and one identical sub crop. Feed the exact
        # official dynamic-HD tensor so the reference path remains independent of our fast path.
        b = pixels.shape[0]
        normalized = pixels * 2.0 - 1.0
        image_pixels = torch.stack([normalized, normalized], dim=1)
        image_sizes = torch.full((b, 2), 448, dtype=torch.long, device=device)
        image_mask = torch.ones(b, 2, 32, 32, dtype=torch.long, device=device)
        image_ids = torch.full(
            (b, Phi4MMVisionEmbedding.num_tokens),
            IMAGE_TOKEN_ID,
            dtype=torch.long,
            device=device,
        )
        image_embeds = model.model.embed_tokens_extend.image_embed(
            input_ids=image_ids,
            input_embeds=image_pixels,
            image_sizes=image_sizes,
            image_attention_mask=image_mask,
            wte=model.model.embed_tokens,
        )

        model.set_lora_adapter("vision")
        multimodal_embeds = torch.cat([image_embeds, model.model.embed_tokens(text_ids)], dim=1)
        vision = model.model(
            inputs_embeds=multimodal_embeds,
            use_cache=False,
            output_hidden_states=False,
        ).last_hidden_state

    meta = {
        "image_embeds": image_embeds.float().cpu(),
        "language": language.float().cpu(),
        "vision": vision.float().cpu(),
    }
    del model, language, vision, image_embeds, multimodal_embeds
    gc.collect()
    torch.cuda.empty_cache()
    return meta


def ours_outputs(
    phi_dir: str,
    text_ids: torch.Tensor,
    pixels: torch.Tensor,
    device: str,
    dtype: torch.dtype,
):
    # The GEN branch is irrelevant to this equivalence test, so shrink only that random expert.
    mot = MoTConfig.from_phi_dir(
        phi_dir,
        gen_hidden_size=128,
        gen_num_attention_heads=8,
        gen_intermediate_size=256,
        enable_action_gen=False,
    )
    model = MoTWorldModel(WorldModelConfig(mot=mot)).to(device=device, dtype=dtype).eval()
    model.load_pretrained()
    text_ids = text_ids.to(device)
    pixels = pixels.to(device=device, dtype=dtype)

    with torch.no_grad():
        text_embeds = model.mot.embed_tokens(text_ids)
        text_pos = torch.arange(text_ids.shape[1], device=device)
        text_pos = text_pos.view(1, 1, -1).expand(3, text_ids.shape[0], -1)
        language, language_kv, _ = model.mot.forward_und(
            text_embeds,
            text_pos,
            use_vision_lora=False,
        )

        image_embeds = model.vision(pixels, encoder_grad=False)
        multimodal_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        mm_pos = torch.arange(multimodal_embeds.shape[1], device=device)
        mm_pos = mm_pos.view(1, 1, -1).expand(3, text_ids.shape[0], -1)
        vision, vision_kv, _ = model.mot.forward_und(
            multimodal_embeds,
            mm_pos,
            use_vision_lora=True,
        )

    out = {
        "image_embeds": image_embeds.float().cpu(),
        "language": language.float().cpu(),
        "vision": vision.float().cpu(),
        "language_kv": len(language_kv),
        "vision_kv": len(vision_kv),
    }
    del model, language, vision, image_embeds, language_kv, vision_kv
    gc.collect()
    torch.cuda.empty_cache()
    return out


def compare(
    name: str,
    ours: torch.Tensor,
    reference: torch.Tensor,
    relative_tol: float,
    cosine_tol: float,
) -> bool:
    abs_err = (ours - reference).abs().max().item()
    rel_err = abs_err / max(reference.abs().max().item(), 1e-12)
    cosine = F.cosine_similarity(ours.flatten(), reference.flatten(), dim=0).item()
    ok = rel_err < relative_tol and cosine > cosine_tol
    print(
        f"[{name:13s}] shape={tuple(ours.shape)}  max|diff|={abs_err:.3e}  "
        f"relative={rel_err:.3e}  cosine={cosine:.8f}  {'OK' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi_dir", default=PHI_DIR)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    dtype = torch.float32 if args.fp32 else torch.bfloat16
    torch.manual_seed(0)
    text_ids = torch.randint(0, 32000, (args.batch, args.seq))
    pixels = torch.rand(args.batch, 3, 448, 448)

    print(f"[setup] dtype={dtype} batch={args.batch} text={args.seq} image_tokens=545")
    reference = reference_outputs(
        args.phi_dir,
        text_ids,
        pixels,
        args.device,
        dtype,
    )
    ours = ours_outputs(args.phi_dir, text_ids, pixels, args.device, dtype)

    # FP32 establishes architectural equivalence. In production BF16, fused SDPA and the
    # checkpoint's eager reference accumulate different rounding over 32 language layers and
    # 27 vision layers, while preserving the direction of the representation.
    relative_tol = 2e-4 if dtype is torch.float32 else 9e-2
    cosine_tol = 0.99999 if dtype is torch.float32 else 0.997
    ok = True
    ok &= compare(
        "image embeds",
        ours["image_embeds"],
        reference["image_embeds"],
        relative_tol,
        cosine_tol,
    )
    ok &= compare(
        "language UND",
        ours["language"],
        reference["language"],
        relative_tol,
        cosine_tol,
    )
    ok &= compare(
        "vision UND",
        ours["vision"],
        reference["vision"],
        relative_tol,
        cosine_tol,
    )
    kv_ok = ours["language_kv"] == ours["vision_kv"] == 32
    ok &= kv_ok
    print(
        f"[per-layer K/V] language={ours['language_kv']} vision={ours['vision_kv']}  "
        f"{'OK' if kv_ok else 'FAIL'}"
    )

    print("\nALL CHECKS PASSED" if ok else "\nFAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
