"""Causal visual context shared by dataset sampling and per-observation rollout buffers."""

from collections import deque

import torch


def visual_context_indices(step, offsets, include_first):
    indices = ([0] if include_first else []) + [max(0, step + offset) for offset in offsets] + [step]
    valid = ([True] if include_first else []) + [step + offset >= 0 for offset in offsets] + [True]
    return indices, valid


class VisualHistoryBuffer:
    def __init__(self, offsets, include_first):
        self.offsets = tuple(offsets)
        self.include_first = include_first
        self.frames = deque(maxlen=1 - min(self.offsets, default=0))
        self.first = None
        self.step = -1
        self.latest = None

    @staticmethod
    def current(batch):
        images = batch.get("images_t0")
        if images is None:
            images = batch["image_t0"][:, None]
        mask = batch.get("image_view_mask")
        if mask is None:
            mask = torch.ones(images.shape[:2], dtype=torch.bool, device=images.device)
        return images, mask.bool()

    def update(self, batch):
        images, mask = self.current(batch)
        frame = (images.detach().clone(), mask.detach().clone())
        if self.first is None:
            self.first = frame
        elif images.shape != self.first[0].shape:
            raise ValueError("Visual history shape changed; reset the episode history buffer.")
        self.step += 1
        self.frames.append(frame)
        indices, valid = visual_context_indices(self.step, self.offsets, self.include_first)
        start = self.step - len(self.frames) + 1
        selected = [self.first if index == 0 else self.frames[index - start] for index in indices]
        self.latest = {
            "context_images": torch.stack([value[0] for value in selected], dim=1),
            "context_image_mask": torch.stack(
                [value[1] & keep for value, keep in zip(selected, valid, strict=True)], dim=1
            ),
            "context_frame_offsets": torch.tensor(indices, device=images.device)[None].expand(
                images.shape[0], -1
            )
            - self.step,
        }
        return self.attach(batch)

    def attach(self, batch):
        if self.latest is None:
            raise ValueError("Update visual history on every observation before planning.")
        images, mask = self.current(batch)
        last_images, last_mask = self.frames[-1]
        if not torch.equal(images.cpu(), last_images.cpu()) or not torch.equal(mask.cpu(), last_mask.cpu()):
            raise ValueError(
                "History does not end at the current observation; update it every control frame."
            )
        return {**batch, **{key: value.to(images.device) for key, value in self.latest.items()}}
