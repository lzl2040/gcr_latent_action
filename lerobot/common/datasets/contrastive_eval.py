"""A fixed evaluation split for the perception <-> physical contrastive model.

The in-batch retrieval accuracy logged during training cannot be used to compare two runs.
Its batches are drawn at random and their *composition* -- which datasets and which episodes
happen to land together -- dominates the number far more than model quality does. Measured
directly: two runs of identical code with the same seed differ by a mean absolute 0.055 in
retrieval accuracy over a 260-step window, which is larger than any code change we have been
trying to measure. Dataloader worker interleaving and video decoding are not deterministic,
so the same seed does not produce the same batches.

This module removes that variance by fixing the frames. It draws a deterministic list of
``(dataset_idx, frame_idx)`` batches once, and every evaluation -- at every step, in every
run -- scores exactly those. What remains is the model.

Two splits are built:

``mixture``
    Sampled with the training weights, with the same hard-negative structure the training
    sampler uses. This is the headline number.
``tactile``
    Sampled only from datasets that carry tactile. In ``debug_research_data`` tactile is 2.7%
    of the mixture, so the headline number would contain about seven tactile rows per batch
    of 256 -- far too few to say anything. This split is entirely tactile, so it can.

Accuracy is computed *within* each batch on a single rank, never across an all-gather. That
keeps the task a fixed N-way retrieval no matter how many GPUs the run used, so numbers stay
comparable across differently-sized runs. With the default 256-frame batches, chance is
1/256 = 0.0039.

Metrics per split:

``acc``
    Retrieval accuracy: for each perception embedding, does its own physical embedding score
    highest among the 256 in the batch? False negatives (same episode, nearby frame) are
    masked out of the candidates first, exactly as in training.
``pos_sim``
    Mean cosine similarity of the matched pair. Both towers are L2-normalised, so this is in
    [-1, 1].
``neg_sim``
    Mean cosine similarity over the negatives that actually enter the InfoNCE denominator.
``gap``
    ``pos_sim - neg_sim``. **Read this, not pos_sim.** ``pos_sim`` on its own cannot tell a
    model that has learned to align the two towers from one that maps every input to nearly
    the same direction -- both drive it towards 1. Only the gap separates them.
``erank_perception`` / ``erank_physical``
    Entropy-based effective rank of each tower's embeddings, on the same scale as
    ``projection_dim`` (512). Detects dimensional collapse, which ``acc`` can hide: measured
    on synthetic data, ``pos_sim`` around 0.98 corresponds to an effective rank near 1, i.e.
    the embeddings occupy a single direction plus a thin residual.

A caveat worth stating plainly: these frames are drawn from the training mixture, not from a
withheld set of episodes. At the scale these runs reach (~0.6 epoch over 36M frames) the
chance that a given eval frame was ever trained on is a couple of percent, so it is
effectively held out -- but it is not held out by construction, and after several epochs this
would stop being true.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler

from lerobot.common.datasets.contrastive_sampler import ContrastiveBatchSampler


class _FixedBatchSampler(Sampler):
    """Yields a pre-computed, immutable list of batches."""

    def __init__(self, batches: list[list[tuple[int, int]]]):
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _draw_batches(
    dataset,
    weights: np.ndarray,
    batch_size: int,
    num_batches: int,
    horizon: int | list[int],
    seed: int,
    policy_cfg,
) -> list[list[tuple[int, int]]]:
    """Draw ``num_batches`` batches with the training sampler's negative structure."""
    sampler = ContrastiveBatchSampler(
        episode_ranges=dataset.episode_ranges,
        sample_weights=weights,
        batch_size=batch_size,
        num_replicas=1,
        rank=0,
        seed=seed,
        samples_per_epoch=batch_size * num_batches,
        horizon=horizon,
        same_dataset_frac=policy_cfg.same_dataset_frac,
        episode_group_frac=policy_cfg.episode_group_frac,
        episode_group_size=policy_cfg.episode_group_size,
        min_frame_gap=policy_cfg.min_frame_gap,
    )
    # A dedicated epoch id keeps these batches disjoint in the RNG stream from anything the
    # training sampler draws, so an eval batch is not accidentally a training batch.
    sampler.set_epoch(9_999_999)
    return [list(b) for b in sampler]


def build_eval_loaders(
    dataset,
    policy_cfg,
    collate_fn,
    batch_size: int = 256,
    num_batches: int = 8,
    num_workers: int = 4,
    seed: int = 100_003,
    rank: int = 0,
    world_size: int = 1,
) -> dict[str, DataLoader]:
    """Build the fixed ``mixture`` and ``tactile`` eval loaders.

    Every rank draws the same batch list and then takes a disjoint stride of it, so the split
    is a fixed set of frames regardless of how many GPUs run it, and doubling the GPUs halves
    the wall time instead of scoring everything twice.

    Returns a possibly-empty dict; the ``tactile`` split is skipped when the mixture has no
    tactile datasets.
    """
    horizon = dataset.frame_horizons
    loaders: dict[str, DataLoader] = {}

    splits: dict[str, np.ndarray] = {"mixture": np.asarray(dataset.sample_weights, dtype=np.float64)}
    tactile_weights = np.asarray(dataset.sample_weights, dtype=np.float64) * dataset.has_tactile
    if tactile_weights.sum() > 0:
        splits["tactile"] = tactile_weights / tactile_weights.sum()

    for offset, (name, weights) in enumerate(splits.items()):
        batches = _draw_batches(
            dataset, weights, batch_size, num_batches, horizon, seed + offset, policy_cfg
        )
        shard = batches[rank::world_size]
        loaders[name] = DataLoader(
            dataset=dataset,
            batch_sampler=_FixedBatchSampler(shard),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            # Not persistent: eval runs rarely, and holding a second set of workers alive for
            # the whole run costs memory the training loader could use.
            persistent_workers=False,
        )
    return loaders


@torch.no_grad()
def _effective_rank(gram: torch.Tensor) -> float:
    """Entropy-based effective rank of a second-moment matrix.

    ``exp(H(p))`` where ``p`` is the eigenvalue spectrum normalised to sum to 1. Equals ``d``
    when the embeddings spread evenly over all ``d`` directions and ``1`` when they collapse
    onto a single one, so it reads on the same scale as ``projection_dim``.
    """
    ev = torch.linalg.eigvalsh(gram.double().cpu()).clamp(min=0.0)
    total = ev.sum()
    if total <= 0:
        return 0.0
    p = ev / total
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


@torch.no_grad()
def evaluate(model_engine, loaders: dict[str, DataLoader], move_batch) -> dict[str, float]:
    """Score every fixed batch and return flat ``eval/<split>_<metric>`` values.

    ``move_batch`` is supplied by the caller so that eval batches are placed and cast exactly
    as training batches are -- the model runs in bf16, and a plain ``.to(device)`` would feed
    it fp32.
    """
    device = model_engine.device
    was_training = model_engine.training
    model_engine.eval()
    policy = model_engine.module if hasattr(model_engine, "module") else model_engine
    metrics: dict[str, float] = {}

    for name, loader in loaders.items():
        hits = torch.zeros((), device=device, dtype=torch.float64)
        rows = torch.zeros((), device=device, dtype=torch.float64)
        pos_sim = torch.zeros((), device=device, dtype=torch.float64)
        neg_sim = torch.zeros((), device=device, dtype=torch.float64)
        neg_pairs = torch.zeros((), device=device, dtype=torch.float64)
        gram_p: torch.Tensor | None = None
        gram_r: torch.Tensor | None = None

        for batch in loader:
            batch = move_batch(batch, device)
            perception, _, _ = policy.encode_perception(batch)
            physical, _ = policy.encode_physical(batch)

            n = perception.shape[0]
            sim = perception @ physical.t()
            episode_uid = batch["episode_uid"].to(device).long().reshape(-1)
            frame_index = batch["frame_index"].to(device).long().reshape(-1)
            labels = torch.arange(n, device=device)
            invalid = policy._false_negative_mask(
                episode_uid, frame_index, episode_uid, frame_index
            )
            invalid[labels, labels] = False
            logits = sim.masked_fill(invalid, float("-inf"))

            hits += (logits.argmax(dim=-1) == labels).sum().double()
            rows += n
            pos_sim += (perception * physical).sum(-1).sum().double()

            # The negatives that actually enter the InfoNCE denominator: off-diagonal and not
            # discarded as a false negative. `pos_sim` alone cannot distinguish a model that
            # aligns the two towers from one that maps everything to a single direction --
            # both give pos_sim -> 1. The gap against `neg_sim` is what separates them.
            competing = ~invalid
            competing[labels, labels] = False
            neg_sim += sim[competing].sum().double()
            neg_pairs += competing.sum().double()

            # Second-moment matrices, accumulated to report effective rank below. Dimensional
            # collapse (embeddings spanning far fewer than `projection_dim` directions) can
            # coexist with a healthy pos/neg gap, so it is worth measuring separately.
            p64, r64 = perception.double(), physical.double()
            gram_p = p64.t() @ p64 if gram_p is None else gram_p + p64.t() @ p64
            gram_r = r64.t() @ r64 if gram_r is None else gram_r + r64.t() @ r64

        if dist.is_initialized():
            stacked = torch.stack([hits, rows, pos_sim, neg_sim, neg_pairs])
            dist.all_reduce(stacked)
            hits, rows, pos_sim, neg_sim, neg_pairs = (stacked[i] for i in range(5))
            if gram_p is not None:
                dist.all_reduce(gram_p)
                dist.all_reduce(gram_r)

        denom = max(rows.item(), 1.0)
        mean_pos = pos_sim.item() / denom
        mean_neg = neg_sim.item() / max(neg_pairs.item(), 1.0)
        metrics[f"eval/{name}_acc"] = hits.item() / denom
        metrics[f"eval/{name}_pos_sim"] = mean_pos
        metrics[f"eval/{name}_neg_sim"] = mean_neg
        metrics[f"eval/{name}_gap"] = mean_pos - mean_neg
        metrics[f"eval/{name}_rows"] = rows.item()
        if gram_p is not None:
            metrics[f"eval/{name}_erank_perception"] = _effective_rank(gram_p)
            metrics[f"eval/{name}_erank_physical"] = _effective_rank(gram_r)

    if was_training:
        model_engine.train()
    return metrics
