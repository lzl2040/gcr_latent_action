"""Score a frozen tactile encoder on data it may never have seen.

Two questions come up whenever the tactile stack changes, and neither is answerable by
looking at the loss curve, because the tactile stream is a minority of the batch and its gate
starts closed:

* Does the encoder *respond* to this sensor at all, or does it emit a near-constant vector?
* Is what it emits *structured*, or is it noise?

This script answers both without training anything, by embedding frames sampled inside
episodes and scoring the embeddings on three unsupervised proxies:

``spread``
    Mean pairwise cosine distance. An encoder that has collapsed on this sensor drives it to
    zero regardless of how large the feature norm is.
``rho``
    Spearman correlation between feature distance and frame distance, within an episode.
    Contact evolves continuously, so an embedding that tracks it grows steadily apart as
    frames get further apart. Around zero means the feature is not following the contact.
``auc``
    P(a within-episode pair is closer than a cross-episode pair). 0.5 is no structure at all.
    Read it with care: episode identity is partly gel wear and lighting, which is structure
    the encoder can pick up without understanding contact, so treat a high ``auc`` with a low
    ``rho`` as suspicious rather than good.

These are proxies, not task performance. They are reliable for detecting *absence* of signal
and much weaker as a ranking between two encoders that both work.

Usage
-----
Compare the FTP-1 ViT against the ImageNet ResNet on every dataset that is local::

    python scripts/tactile_feature_probe.py --dataset ftp_1_sharpa=/Data/.../FTP-1/sharpa

Repeat ``--dataset NAME=ROOT`` for as many as you want. ``NAME`` is the registry name, used
to look up which FTP-1 sensor tokenizer and z-score to dispatch to.

Note on sampling: frames are spread over the whole episode on purpose. Episodes open with a
pre-contact idle stretch where the gel is constant (sharpa's first 96 frames have a pixel std
of 3e-4), so sampling a prefix scores the idle period and every encoder looks collapsed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FTP1_SIZE = 224


def tactile_image_keys(info: dict) -> list[str]:
    return sorted(
        k
        for k, v in info["features"].items()
        if "tactile" in k and v.get("dtype") in ("video", "image")
    )


def episodes(root: Path, info: dict, key: str, n_ep: int, n_fr: int) -> list[torch.Tensor]:
    from torchcodec.decoders import VideoDecoder

    template = info["video_path"]
    if "{video_key}" in template and "file_index" in template:
        files = sorted((root / "videos" / key).rglob("*.mp4"))
    else:
        files = sorted((root / "videos").rglob(f"{key}/*.mp4"))

    out = []
    for f in files[:n_ep]:
        try:
            dec = VideoDecoder(str(f), device="cpu")
            total = dec.metadata.num_frames
            if total < n_fr:
                continue
            idx = sorted({int(i) for i in np.linspace(0, total - 1, n_fr)})
            x = dec.get_frames_at(indices=idx).data.float()
        except Exception as exc:  # noqa: BLE001
            logger.debug("  skipping %s: %s", f.name, exc)
            continue
        out.append(F.interpolate(x, size=(FTP1_SIZE, FTP1_SIZE), mode="bilinear", align_corners=False))
    return out


def score(feats_per_ep: list[torch.Tensor]) -> tuple[float, float, float]:
    from scipy.stats import spearmanr

    f = F.normalize(torch.cat(feats_per_ep).float(), dim=-1)
    dist = (1 - f @ f.T).numpy()
    spread = float(dist.mean())

    rhos, off = [], 0
    for e in feats_per_ep:
        k = len(e)
        d = dist[off : off + k, off : off + k]
        off += k
        t = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :]).astype(float)
        iu = np.triu_indices(k, 1)
        if d[iu].std() > 1e-9:
            rhos.append(spearmanr(d[iu], t[iu]).statistic)

    ep_id = np.concatenate([np.full(len(e), i) for i, e in enumerate(feats_per_ep)])
    same = ep_id[:, None] == ep_id[None, :]
    iu = np.triu_indices(len(f), 1)
    within, cross = dist[iu][same[iu]], dist[iu][~same[iu]]
    # P(within < cross), estimated exactly rather than by sampling; both sides are small.
    auc = float((within[:, None] < cross[None, :]).mean()) if len(within) and len(cross) else 0.5
    return spread, float(np.mean(rhos)) if rhos else 0.0, auc


def embed_ftp1(eps, name: str, n_keys: int, weights_dir: str, device: str):
    from lerobot.common.policies.ace.ftp1_tactile import (
        FTP1_SENSOR_NAMES,
        FTP1TactileTower,
        tactile_image_sensors,
    )

    ids, means, stds = tactile_image_sensors(name, n_keys)
    sensor = FTP1_SENSOR_NAMES[ids[0]]
    tower = FTP1TactileTower(weights_dir, [sensor]).eval().to(device)
    m = torch.tensor(means[0], dtype=torch.float32).view(1, 3).to(device)
    s = torch.tensor(stds[0], dtype=torch.float32).view(1, 3).to(device)
    out = []
    for e in eps:
        e = e.clamp(0, 255).to(torch.uint8).to(device)
        n = len(e)
        sid = torch.full((n,), ids[0], device=device)
        with torch.no_grad():
            out.append(tower(e.unsqueeze(1), sid, m.expand(n, 3), s.expand(n, 3))[:, 0].cpu())
    return out, sensor


def embed_resnet(eps, device: str):
    import torchvision

    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.eval().to(device)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    out = []
    for e in eps:
        x = (e.to(device) / 255.0 - mean) / std
        with torch.no_grad():
            out.append(model(x).cpu())
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", action="append", required=True, metavar="NAME=ROOT",
                   help="registry name and dataset root, repeatable")
    p.add_argument("--ftp1-dir", default="/Data/lzl/huggingface/ftp1_v0426_50kstep")
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--skip-ftp1", action="store_true", help="only score the ImageNet ResNet")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    header = f"{'dataset':<34s} {'sensor':<16s} |"
    if not args.skip_ftp1:
        header += f" {'FTP-1 ViT':^22s} |"
    header += f" {'ImageNet ResNet18':^22s}"
    logger.info(header)
    sub = f"{'':<34s} {'':<16s} |"
    cols = f"{'spread':>7s}{'rho':>7s}{'auc':>8s}"
    logger.info(sub + (f" {cols} |" if not args.skip_ftp1 else "") + f" {cols}")

    for spec in args.dataset:
        if "=" not in spec:
            raise SystemExit(f"--dataset expects NAME=ROOT, got {spec!r}")
        name, root = spec.split("=", 1)
        root = Path(root)
        if not (root / "meta" / "info.json").exists():
            logger.warning("%-34s -- no meta/info.json at %s", name, root)
            continue
        info = json.loads((root / "meta" / "info.json").read_text())
        keys = tactile_image_keys(info)
        if not keys:
            logger.warning("%-34s -- no tactile image keys", name)
            continue
        eps = episodes(root, info, keys[0], args.episodes, args.frames)
        if len(eps) < 2:
            logger.warning("%-34s -- fewer than 2 usable episodes", name)
            continue

        row, sensor = f"{name:<34s} ", "-"
        parts = []
        if not args.skip_ftp1:
            feats, sensor = embed_ftp1(eps, name, len(keys), args.ftp1_dir, args.device)
            a = score(feats)
            parts.append(f"{a[0]:7.4f}{a[1]:7.3f}{a[2]:8.3f}")
        b = score(embed_resnet(eps, args.device))
        parts.append(f"{b[0]:7.4f}{b[1]:7.3f}{b[2]:8.3f}")
        logger.info(row + f"{sensor:<16s} | " + " | ".join(parts))


if __name__ == "__main__":
    main()
