import random
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from lerobot.common.datasets.b1k_local_dataset import (
    B1KLocalTemporalDataset,
    B1KSourceConfig,
    behavior_source_configs,
    resolve_task_episode_ordinals,
    split_validation_ordinals,
)


class B1KSARMSequenceDataset(Dataset):
    """B1K temporal-window adapter that emits SARM-compatible samples."""

    def __init__(
        self,
        *,
        behavior_repo_root: str | None = None,
        source: B1KSourceConfig,
        camera_names: list[str],
        n_obs_steps: int,
        frame_gap: int,
        max_rewind_steps: int,
        seed: int,
        chunk_streaming_using_keyframe: bool = True,
        temporal_all_windows_per_chunk: bool = False,
        use_dtw_progress: bool = False,
        image_size: int = 224,
        eval_frame_gap: int | None = None,
        stream_video_decode: bool = True,
        video_chunk_size: int = 250,
        video_cache_chunks: int = 4,
    ) -> None:
        del behavior_repo_root
        self.source = source
        self.camera_names = list(camera_names)
        self.n_obs_steps = int(n_obs_steps)
        self.frame_gap = int(frame_gap)
        self.max_rewind_steps = int(max_rewind_steps)
        self.sequence_len = self.n_obs_steps + 1
        self.use_dtw_progress = bool(use_dtw_progress)
        self.image_size = int(image_size)
        self.dataset = B1KLocalTemporalDataset(
            source=source,
            camera_names=camera_names,
            n_obs_steps=n_obs_steps,
            frame_gap=frame_gap,
            seed=seed,
            chunk_streaming_using_keyframe=chunk_streaming_using_keyframe,
            temporal_all_windows_per_chunk=temporal_all_windows_per_chunk,
            eval_frame_gap=eval_frame_gap,
            stream_video_decode=stream_video_decode,
            video_chunk_size=video_chunk_size,
            video_cache_chunks=video_cache_chunks,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _resize_single_image_with_pad_pil(image: np.ndarray, height: int, width: int) -> np.ndarray:
        original_dtype = image.dtype
        float_range = None
        if np.issubdtype(original_dtype, np.floating):
            image_min = float(np.nanmin(image)) if image.size else 0.0
            if image_min < 0.0:
                float_range = "minus_one_to_one"
                pil_array = np.clip((image + 1.0) * 127.5, 0, 255).astype(np.uint8)
            else:
                float_range = "zero_to_one"
                pil_array = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            pil_array = image.astype(np.uint8, copy=False)

        channels = pil_array.shape[-1]
        pil_image = Image.fromarray(pil_array[..., 0] if channels == 1 else pil_array)
        cur_width, cur_height = pil_image.size
        if cur_width == width and cur_height == height:
            resized_array = pil_array
        else:
            ratio = max(cur_width / width, cur_height / height)
            resized_height = int(cur_height / ratio)
            resized_width = int(cur_width / ratio)
            resized_image = pil_image.resize((resized_width, resized_height), resample=Image.BILINEAR)
            canvas = Image.new(resized_image.mode, (width, height), 0)
            pad_height = max(0, int((height - resized_height) / 2))
            pad_width = max(0, int((width - resized_width) / 2))
            canvas.paste(resized_image, (pad_width, pad_height))
            resized_array = np.asarray(canvas)
            if channels == 1:
                resized_array = resized_array[..., None]

        if float_range == "minus_one_to_one":
            return (resized_array.astype(np.float32) / 127.5 - 1.0).astype(original_dtype)
        if float_range == "zero_to_one":
            return (resized_array.astype(np.float32) / 255.0).astype(original_dtype)
        return resized_array.astype(original_dtype, copy=False)

    def _image_to_chw_tensor(self, image: Any) -> torch.Tensor:
        arr = B1KSARMSequenceDataset._to_numpy(image)
        if arr.ndim != 4:
            raise ValueError(f"Expected temporal image [T,H,W,C] or [T,C,H,W], got shape={arr.shape}")
        if arr.shape[-1] in (1, 3, 4):
            arr = np.moveaxis(arr, -1, 1)
        arr = np.moveaxis(arr, 1, -1)
        if arr.shape[-3:-1] != (self.image_size, self.image_size):
            arr = np.stack(
                [
                    self._resize_single_image_with_pad_pil(frame, self.image_size, self.image_size)
                    for frame in arr
                ],
                axis=0,
            )
        arr = np.moveaxis(arr, -1, 1)
        if arr.dtype == np.uint8:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)
            if arr.max(initial=0.0) > 2.0:
                arr = arr / 255.0
        return torch.from_numpy(arr[:, :3])

    def _append_rewind(self, result: dict[str, Any]) -> dict[str, Any]:
        if self.max_rewind_steps <= 0:
            return result
        valid_len = int(result["lengths"].item())
        rewind_steps = random.randint(0, self.max_rewind_steps)
        if rewind_steps > 0:
            src_indices = list(range(max(valid_len - 2, 0), max(valid_len - rewind_steps - 2, -1), -1))
            src_indices = src_indices[:rewind_steps]
            if src_indices:
                for cam in self.camera_names:
                    result[cam] = torch.cat([result[cam], result[cam][src_indices]], dim=0)
                for key in (
                    "state",
                    "targets",
                    "total_progress",
                    "stage_token_index",
                    "stage_sum",
                    "ep_idx",
                    "frame_idx",
                    "reward",
                    "q_score",
                ):
                    result[key] = torch.cat([result[key], result[key][src_indices]], dim=0)
                result["frame_relative_indices"] = torch.cat(
                    [result["frame_relative_indices"], result["frame_relative_indices"][src_indices]],
                    dim=0,
                )
                result["lengths"] = torch.tensor(valid_len + len(src_indices), dtype=torch.int32)
        return self._pad(result)

    def _pad(self, result: dict[str, Any]) -> dict[str, Any]:
        target_len = self.sequence_len + self.max_rewind_steps
        pad = target_len - int(result["targets"].shape[0])
        if pad <= 0:
            return result
        for cam in self.camera_names:
            result[cam] = torch.cat(
                [result[cam], torch.zeros((pad, *result[cam].shape[1:]), dtype=result[cam].dtype)],
                dim=0,
            )
        result["state"] = torch.cat(
            [result["state"], torch.zeros((pad, result["state"].shape[-1]), dtype=result["state"].dtype)],
            dim=0,
        )
        for key, dtype in (
            ("targets", torch.float32),
            ("total_progress", torch.float32),
            ("stage_token_index", torch.int64),
            ("stage_sum", torch.int64),
            ("ep_idx", torch.int64),
            ("frame_idx", torch.int64),
            ("reward", torch.float32),
            ("q_score", torch.float32),
        ):
            if key in result:
                result[key] = torch.cat([result[key], torch.zeros((pad,), dtype=dtype)], dim=0)
        result["frame_relative_indices"] = torch.cat(
            [result["frame_relative_indices"], torch.ones((pad,), dtype=torch.float32)],
            dim=0,
        )
        return result

    def __getitem__(self, idx: int) -> dict[str, Any]:
        transformed = self.dataset[idx]
        state = torch.as_tensor(self._to_numpy(transformed["state"]), dtype=torch.float32)
        stage = torch.as_tensor(self._to_numpy(transformed["stage_token_index"]), dtype=torch.long)
        stage_sum = torch.as_tensor(self._to_numpy(transformed["stage_sum"]), dtype=torch.long)
        inner = torch.as_tensor(self._to_numpy(transformed["progress_margin_target"]), dtype=torch.float32)
        total_progress = torch.as_tensor(self._to_numpy(transformed["progress_target"]), dtype=torch.float32)
        if self.use_dtw_progress:
            inner = torch.clamp(total_progress * 10.0 - stage.float(), 0.0, 0.999)
        targets = stage.float() + torch.clamp(inner, 0.0, 0.999)

        result: dict[str, Any] = {
            "targets": targets,
            "total_progress": total_progress,
            "lengths": torch.tensor(self.sequence_len, dtype=torch.int32),
            "task": self.source.task_name,
            "state": state,
            "stage_token_index": stage,
            "stage_sum": stage_sum,
            "frame_relative_indices": torch.linspace(0.0, 1.0, steps=self.sequence_len, dtype=torch.float32),
            "ep_idx": torch.as_tensor(self._to_numpy(transformed["ep_idx"]), dtype=torch.int64),
            "frame_idx": torch.as_tensor(self._to_numpy(transformed["frame_idx"]), dtype=torch.int64),
            "reward": torch.as_tensor(self._to_numpy(transformed.get("reward", np.zeros(self.sequence_len))), dtype=torch.float32),
            "q_score": torch.as_tensor(self._to_numpy(transformed.get("q_score", np.ones(self.sequence_len))), dtype=torch.float32),
            "source_name": self.source.name,
        }

        image_key_by_camera = {
            "observation.images.rgb.head": "base_0_rgb",
            "observation.images.rgb.left_wrist": "left_wrist_0_rgb",
            "observation.images.rgb.right_wrist": "right_wrist_0_rgb",
        }
        for cam in self.camera_names:
            result[cam] = self._image_to_chw_tensor(transformed["image"][image_key_by_camera[cam]])

        return self._append_rewind(result)


class B1KMixtureDataset(Dataset):
    def __init__(self, datasets: list[Dataset], sample_weights: list[float] | None = None, seed: int = 42):
        if not datasets:
            raise ValueError("B1KMixtureDataset requires at least one dataset.")
        self.datasets = datasets
        weights = sample_weights or [1.0 / len(datasets)] * len(datasets)
        if len(weights) != len(datasets):
            raise ValueError("sample_weights length must match datasets.")
        total = float(sum(weights))
        self.sample_weights = [float(w) / total for w in weights]
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return max(len(ds) for ds in self.datasets)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx = self.rng.choices(range(len(self.datasets)), weights=self.sample_weights, k=1)[0]
        ds = self.datasets[source_idx]
        return ds[idx % len(ds)]


class B1KSequentialDataset(Dataset):
    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.offsets = []
        offset = 0
        for ds in datasets:
            self.offsets.append(offset)
            offset += len(ds)
        self.total_len = offset

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> dict[str, Any]:
        for ds, offset in zip(self.datasets, self.offsets):
            if idx < offset + len(ds):
                return ds[idx - offset]
        raise IndexError(idx)


def collate_b1k_sarm(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "targets": torch.stack([b["targets"] for b in batch]),
        "total_progress": torch.stack([b["total_progress"] for b in batch]),
        "lengths": torch.stack([b["lengths"] for b in batch]),
        "task": [b["task"] for b in batch],
        "state": torch.stack([b["state"] for b in batch]),
        "stage_token_index": torch.stack([b["stage_token_index"] for b in batch]),
        "stage_sum": torch.stack([b["stage_sum"] for b in batch]),
        "frame_relative_indices": torch.stack([b["frame_relative_indices"] for b in batch]),
        "ep_idx": torch.stack([b["ep_idx"] for b in batch]),
        "frame_idx": torch.stack([b["frame_idx"] for b in batch]),
        "reward": torch.stack([b["reward"] for b in batch]),
        "q_score": torch.stack([b["q_score"] for b in batch]),
        "source_name": [b["source_name"] for b in batch],
    }
    camera_names = [k for k in batch[0] if k.startswith("observation.images.rgb.")]
    for cam in camera_names:
        out[cam] = torch.stack([b[cam] for b in batch])
    return out
