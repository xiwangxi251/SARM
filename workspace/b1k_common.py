import json
import os
import dataclasses
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm

from lerobot.common.datasets.b1k_sarm_dataset import (
    B1KMixtureDataset,
    B1KSARMSequenceDataset,
    B1KSequentialDataset,
    behavior_source_configs,
    collate_b1k_sarm,
)
from lerobot.common.datasets.b1k_local_dataset import _episode_ids_for_source, _load_q_scores
from models.clip_encoder import FrozenCLIPEncoder
from utils.train_utils import save_ckpt, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"


TASK_TEXT = {
    "turning_on_radio": "turn on the radio",
    "picking_up_trash": "pick up the trash and place it in the bin",
}


_PRINTED_SPLITS: set[tuple[str, bool]] = set()


def infinite_loader(dl):
    while True:
        for batch in dl:
            yield batch


def wandb_run(cfg):
    if not bool(getattr(cfg.general, "use_wandb", True)):
        return nullcontext(None)
    import wandb

    return wandb.init(
        project=f"{cfg.general.project_name}-{cfg.general.task_name}",
        name=datetime.now().strftime("%Y.%m.%d-%H.%M.%S"),
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def wandb_log(cfg, data: dict[str, Any], step: int) -> None:
    if bool(getattr(cfg.general, "use_wandb", True)):
        import wandb

        wandb.log(data, step=step)


def make_b1k_datasets(cfg, *, for_eval: bool = False):
    train_sources, val_sources = behavior_source_configs(
        task_name=cfg.general.task_name,
        expert_root=cfg.general.expert_root,
        rft_root=cfg.general.rft_root,
        validation_seed=cfg.general.validation_split_seed,
        validation_ratio=cfg.general.validation_episode_ratio,
    )
    maybe_print_b1k_split(cfg, train_sources, val_sources, for_eval=for_eval)
    if for_eval:
        sources = [
            dataclasses.replace(source, use_q_score_1_only=False)
            if bool(getattr(cfg.eval, "include_failed_episodes", True))
            else source
            for source in val_sources
        ]
    else:
        sources = train_sources
    all_windows = False if for_eval else bool(getattr(cfg.b1k, "temporal_all_windows_per_chunk", False))
    eval_frame_gap = int(getattr(cfg.eval, "eval_frame_gap", 10)) if for_eval else None
    datasets = [
        B1KSARMSequenceDataset(
            behavior_repo_root=getattr(cfg.general, "behavior_repo_root", None),
            source=source,
            camera_names=list(cfg.general.camera_names),
            n_obs_steps=cfg.model.n_obs_steps,
            frame_gap=cfg.model.frame_gap,
            max_rewind_steps=cfg.model.max_rewind_steps if not for_eval else 0,
            seed=cfg.general.seed,
            chunk_streaming_using_keyframe=cfg.b1k.chunk_streaming_using_keyframe,
            temporal_all_windows_per_chunk=all_windows,
            use_dtw_progress=cfg.b1k.use_dtw_progress,
            image_size=getattr(cfg.model, "image_size", 224),
            eval_frame_gap=eval_frame_gap,
        )
        for source in sources
    ]
    if for_eval:
        return B1KSequentialDataset(datasets)

    val_datasets = [
        B1KSARMSequenceDataset(
            behavior_repo_root=getattr(cfg.general, "behavior_repo_root", None),
            source=source,
            camera_names=list(cfg.general.camera_names),
            n_obs_steps=cfg.model.n_obs_steps,
            frame_gap=cfg.model.frame_gap,
            max_rewind_steps=0,
            seed=cfg.general.seed,
            chunk_streaming_using_keyframe=cfg.b1k.chunk_streaming_using_keyframe,
            temporal_all_windows_per_chunk=bool(getattr(cfg.b1k, "temporal_all_windows_per_chunk", False)),
            use_dtw_progress=cfg.b1k.use_dtw_progress,
            image_size=getattr(cfg.model, "image_size", 224),
        )
        for source in val_sources
    ]
    train = B1KMixtureDataset(datasets, sample_weights=list(cfg.general.sample_weights), seed=cfg.general.seed)
    val = B1KSequentialDataset(val_datasets)
    return train, val


def maybe_print_b1k_split(cfg, train_sources, val_sources, *, for_eval: bool) -> None:
    key = (str(cfg.general.task_name), bool(for_eval))
    if key in _PRINTED_SPLITS:
        return
    _PRINTED_SPLITS.add(key)

    print(
        "[B1K Split] "
        f"task={cfg.general.task_name} seed={cfg.general.validation_split_seed} "
        f"ratio={cfg.general.validation_episode_ratio} "
        "behavior_seed_rule=seed+dataset_index*10000+task_offset"
    )
    for split_name, sources in (("train", train_sources), ("val", val_sources)):
        for source in sources:
            root = Path(source.root)
            q_scores = _load_q_scores(root)
            selected_before_q = _episode_ids_for_source(
                root,
                source.task_name,
                source.episodes,
                q_scores,
                False,
            )
            episode_ids = _episode_ids_for_source(
                root,
                source.task_name,
                source.episodes,
                q_scores,
                source.use_q_score_1_only,
            )
            print(
                "[B1K Split] "
                f"{split_name}/{source.name}: count={len(episode_ids)} "
                f"q_score_1_only={source.use_q_score_1_only} "
                f"selected_before_q={selected_before_q} "
                f"episodes={episode_ids}"
            )


def make_loaders(cfg):
    train_ds, val_ds = make_b1k_datasets(cfg)
    return (
        torch.utils.data.DataLoader(train_ds, collate_fn=collate_b1k_sarm, **cfg.dataloader),
        torch.utils.data.DataLoader(val_ds, collate_fn=collate_b1k_sarm, **cfg.val_dataloader),
    )


def make_scheduler(optimizer, cfg):
    warmup = LinearLR(
        optimizer,
        start_factor=1e-6 / cfg.optim.lr,
        end_factor=1.0,
        total_iters=cfg.optim.warmup_steps,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, cfg.optim.total_steps - cfg.optim.warmup_steps),
        eta_min=0.0,
    )
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.optim.warmup_steps])


def encode_batch(batch, cfg, camera_names, device, clip_encoder):
    batch_size, timesteps = batch[camera_names[0]].shape[:2]
    images = []
    for camera in camera_names:
        images.append(batch[camera].flatten(0, 1).to(device))
    state = batch["state"].to(device)
    lengths = batch["lengths"].to(device).long()
    if bool(cfg.model.no_state):
        state = torch.zeros_like(state)

    with torch.no_grad():
        image_emb = clip_encoder.encode_image(torch.cat(images, dim=0))
        image_emb = image_emb.view(len(images), batch_size, timesteps, -1).permute(1, 0, 2, 3)
        text = [TASK_TEXT.get(task, task.replace("_", " ")) for task in batch["task"]]
        lang_emb = clip_encoder.encode_text(text)
    return image_emb, lang_emb, state, lengths


def sequence_mask(lengths, width):
    return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


def masked_mse(pred, target, lengths):
    mask = sequence_mask(lengths, pred.shape[1]).float()
    return (((pred - target) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def masked_cross_entropy(logits, target, lengths):
    batch_size, timesteps, classes = logits.shape
    loss = F.cross_entropy(logits.reshape(batch_size * timesteps, classes), target.reshape(-1), reduction="none")
    mask = sequence_mask(lengths, timesteps).reshape(-1).float()
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def episode_series_to_frame_dict(series: dict[str, dict[str, float]]) -> dict[str, float]:
    out = {}
    for ep, values in series.items():
        for frame, value in values.items():
            out[f"{ep}/{frame}"] = value
    return out


def compute_advantages(expected_series, *, n_step: int, target_positive_ratio: float):
    advantages: dict[str, dict[str, float]] = {}
    masks: dict[str, dict[str, int]] = {}
    all_adv = []
    for ep, values in expected_series.items():
        frames = sorted((int(k), float(v)) for k, v in values.items())
        advantages[ep] = {}
        masks[ep] = {}
        for idx, (frame, value) in enumerate(frames):
            future = [v for _, v in frames[idx + 1 : idx + 1 + n_step]]
            adv = (sum(future) / len(future) - value) if future else 0.0
            advantages[ep][str(frame)] = adv
            all_adv.append(adv)

    if all_adv:
        sorted_adv = sorted(all_adv, reverse=True)
        cutoff_idx = min(len(sorted_adv) - 1, max(0, int(len(sorted_adv) * target_positive_ratio) - 1))
        threshold = sorted_adv[cutoff_idx]
    else:
        threshold = 0.0

    positive = 0
    total = 0
    for ep, values in advantages.items():
        for frame, adv in values.items():
            mask = int(adv >= threshold)
            masks[ep][frame] = mask
            positive += mask
            total += 1

    stats = {
        "num_items": total,
        "positive": positive,
        "positive_ratio": (positive / total) if total else 0.0,
        "threshold": threshold,
        "n_step": n_step,
        "target_positive_ratio": target_positive_ratio,
    }
    return advantages, masks, stats


def series_metadata(series: dict[str, dict[str, float]], cfg) -> dict[str, Any]:
    episodes = sorted(series.keys(), key=lambda item: int(item))
    return {
        "num_episodes": len(episodes),
        "episodes": episodes,
        "num_items": sum(len(values) for values in series.values()),
        "eval_frame_gap": int(getattr(cfg.eval, "eval_frame_gap", 10)),
        "task_name": str(cfg.general.task_name),
        "validation_split_seed": int(cfg.general.validation_split_seed),
        "validation_episode_ratio": float(cfg.general.validation_episode_ratio),
        "include_failed_episodes": bool(getattr(cfg.eval, "include_failed_episodes", True)),
        "note": "B1K export uses the RFT validation split. By default export includes failed episodes even if training filters q_score==1.",
    }


def interpolate_episode_series(series: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    interpolated: dict[str, dict[str, float]] = {}
    for ep, values in series.items():
        points = sorted((int(frame), float(value)) for frame, value in values.items())
        if not points:
            interpolated[ep] = {}
            continue
        if len(points) == 1:
            frame, value = points[0]
            interpolated[ep] = {str(frame): value}
            continue

        ep_values: dict[str, float] = {}
        for (left_frame, left_value), (right_frame, right_value) in zip(points[:-1], points[1:]):
            width = max(right_frame - left_frame, 1)
            for frame in range(left_frame, right_frame):
                alpha = (frame - left_frame) / float(width)
                ep_values[str(frame)] = left_value + alpha * (right_value - left_value)
        last_frame, last_value = points[-1]
        ep_values[str(last_frame)] = last_value
        interpolated[ep] = ep_values
    return interpolated


def save_behavior_value_jsons(output_dir: Path, suffix: str, expected_series, raw_series, cfg):
    suffix = suffix or ""
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        f"expected_values{suffix}.json",
        f"model_raw_values{suffix}.json",
        f"model_raw_values_series{suffix}.json",
        f"advantages{suffix}.json",
        f"advantage_masks{suffix}.json",
    ):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()

    advantages, masks, stats = compute_advantages(
        expected_series,
        n_step=int(cfg.eval.advantage_n_step),
        target_positive_ratio=float(cfg.eval.target_positive_ratio),
    )
    interpolated = interpolate_episode_series(expected_series)

    write_json(output_dir / f"expected_values_series{suffix}.json", expected_series)
    write_json(output_dir / f"expected_values_series{suffix}_interp.json", interpolated)
    write_json(output_dir / f"advantages_series{suffix}.json", advantages)
    write_json(output_dir / f"advantage_masks_series{suffix}.json", masks)
    write_json(output_dir / f"advantage_stats{suffix}.json", stats)
    write_json(output_dir / f"export_metadata{suffix}.json", series_metadata(expected_series, cfg))


class B1KBaseWorkspace:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.general.device if torch.cuda.is_available() else "cpu")
        set_seed(cfg.general.seed)
        self.camera_names = list(cfg.general.camera_names)
        self.save_dir = Path(f"{cfg.general.project_name}/{cfg.general.task_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Init] Using device: {self.device}")
        print(f"[Init] Logging & ckpts to: {self.save_dir}")

    def save_config(self):
        OmegaConf.save(self.cfg, self.save_dir / "config.yaml")
