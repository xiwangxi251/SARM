import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from lerobot.common.datasets.b1k_inputs import B1kInputs


TASK_NAME_TO_INDEX = {
    "turning_on_radio": 0,
    "picking_up_trash": 1,
}

TASK_TEXT_BY_INDEX = {
    0: "Turn on the radio receiver that's on the table in the living room.",
    1: "Pick up the trash on the floor and put it in the trash can.",
}

TASK_PROGRESS_BIN_COUNTS = {
    0: 3,
    1: 4,
}

CAMERA_TO_B1K_KEY = {
    "observation.images.rgb.head": "observation/egocentric_camera",
    "observation.images.rgb.left_wrist": "observation/wrist_image_left",
    "observation.images.rgb.right_wrist": "observation/wrist_image_right",
}


@dataclass(frozen=True)
class B1KSourceConfig:
    name: str
    root: str
    repo_id: str
    task_name: str
    episodes: list[int] | dict[str, list[int]] | None
    use_q_score_1_only: bool = False
    use_for_validation: bool = False
    validation_episode_ratio: float | None = None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_task_episode_ordinals(root: str | Path, task_name: str) -> list[int]:
    root = Path(root).expanduser()
    task_idx = TASK_NAME_TO_INDEX[task_name]
    episodes = [
        int(item["episode_index"])
        for item in _read_jsonl(root / "meta" / "episodes.jsonl")
        if int(item["episode_index"]) // 10000 == task_idx
    ]
    return list(range(len(sorted(episodes))))


def split_validation_ordinals(ordinals: list[int], *, ratio: float, seed: int) -> tuple[list[int], list[int]]:
    if not ordinals:
        raise ValueError("Cannot split an empty episode ordinal list.")
    val_count = max(1, int(len(ordinals) * float(ratio)))
    if val_count >= len(ordinals):
        raise ValueError("Validation split would leave no training episodes.")
    shuffled = list(ordinals)
    random.Random(seed).shuffle(shuffled)
    val = sorted(shuffled[:val_count])
    val_set = set(val)
    train = [idx for idx in ordinals if idx not in val_set]
    return train, val


def behavior_source_configs(
    *,
    task_name: str,
    expert_root: str,
    rft_root: str,
    validation_seed: int = 42,
    validation_ratio: float = 1 / 6,
) -> tuple[list[B1KSourceConfig], list[B1KSourceConfig]]:
    expert_ordinals = resolve_task_episode_ordinals(expert_root, task_name)
    rft_ordinals = resolve_task_episode_ordinals(rft_root, task_name)
    rft_train, rft_val = split_validation_ordinals(rft_ordinals, ratio=validation_ratio, seed=validation_seed)
    rft_q1_only = task_name == "turning_on_radio"
    train = [
        B1KSourceConfig("expert", expert_root, "local/expert", task_name, {task_name: expert_ordinals}),
        B1KSourceConfig("rft", rft_root, "local/rft", task_name, {task_name: rft_train}, rft_q1_only),
    ]
    val = [B1KSourceConfig("rft_val", rft_root, "local/rft", task_name, {task_name: rft_val}, rft_q1_only, True)]
    return train, val


def _load_q_scores(root: Path) -> dict[int, float]:
    path = root / "q_scores_with_episode_ids.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return {int(item["episode_id"]): float(item["q_score"]) for item in payload.get("episodes", [])}


def _episode_ids_for_source(root: Path, task_name: str, episodes: Any, q_scores: dict[int, float], q1_only: bool):
    task_idx = TASK_NAME_TO_INDEX[task_name]
    all_episode_ids = sorted(
        int(item["episode_index"])
        for item in _read_jsonl(root / "meta" / "episodes.jsonl")
        if int(item["episode_index"]) // 10000 == task_idx
    )
    if episodes is None:
        selected = all_episode_ids
    else:
        ordinals = episodes.get(task_name, []) if isinstance(episodes, dict) else episodes
        selected = [all_episode_ids[int(ordinal)] for ordinal in ordinals]
    if q1_only and q_scores:
        selected = [ep for ep in selected if np.isclose(q_scores.get(int(ep), -1.0), 1.0, atol=1e-6)]
    if not selected:
        raise ValueError(f"No episodes selected for task={task_name}, root={root}")
    return selected


def _load_episode_lengths(root: Path) -> dict[int, int]:
    return {int(item["episode_index"]): int(item["length"]) for item in _read_jsonl(root / "meta" / "episodes.jsonl")}


def _data_path(root: Path, task_idx: int, ep_idx: int) -> Path:
    return root / "data" / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.parquet"


def _video_path(root: Path, task_idx: int, ep_idx: int, video_key: str) -> Path:
    return root / "videos" / f"task-{task_idx:04d}" / video_key / f"episode_{ep_idx:08d}.mp4"


def _read_parquet_rows(path: Path, frame_indices: list[int]) -> dict[str, np.ndarray]:
    table = pq.read_table(
        path,
        columns=["index", "episode_index", "task_index", "timestamp", "observation.state"],
    )
    rows = np.asarray(frame_indices, dtype=np.int64)
    return {
        "index": np.asarray(table.column("index").to_numpy())[rows],
        "episode_index": np.asarray(table.column("episode_index").to_numpy())[rows],
        "task_index": np.asarray(table.column("task_index").to_numpy())[rows],
        "timestamp": np.asarray(table.column("timestamp").to_numpy())[rows],
        "observation.state": np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)[rows],
    }


def _decode_video_frames(path: Path, frame_indices: list[int]) -> np.ndarray:
    wanted = set(int(i) for i in frame_indices)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for idx, frame in enumerate(container.decode(stream)):
            if idx in wanted:
                frames[idx] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
    missing = [idx for idx in frame_indices if idx not in frames]
    if missing:
        raise IndexError(f"Missing decoded frames {missing[:5]} from {path}")
    return np.stack([frames[int(idx)] for idx in frame_indices], axis=0)


def _matching_end_frames(entries: list[dict], key: str, allowed: set[tuple[str, ...]], episode_len: int):
    matches = []
    for entry in entries:
        desc = tuple(str(x) for x in entry.get(key, []))
        if desc not in allowed:
            continue
        start, end = entry["frame_duration"]
        end = int(end)
        if end < int(start) or end > episode_len:
            raise ValueError(f"Bad frame_duration={entry['frame_duration']}")
        matches.append((desc, end))
    return matches


def _expert_annotation_spec(root: Path, task_idx: int, ep_idx: int, episode_len: int):
    with (root / "annotations" / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.json").open("r", encoding="utf-8") as f:
        annotation = json.load(f)
    if task_idx == 0:
        del annotation
        grasp = _task0_grasp_frame(root, ep_idx)
        terminal_frame = max(int(episode_len) - 1, 0)
        if grasp < 0 or grasp > terminal_frame:
            raise ValueError(f"Task0 grasp frame out of range: ep={ep_idx}, grasp={grasp}, episode_len={episode_len}")
        return (grasp, terminal_frame), None, int(episode_len)
    if task_idx == 1:
        matches = _matching_end_frames(
            annotation.get("primitive_annotation", []),
            "primitive_description",
            {("pick up from", "place in")},
            episode_len,
        )
        if len(matches) != 3:
            raise ValueError(f"Task1 annotation expects 3 primitives, got {len(matches)}")
        ends = tuple(int(end) for _, end in matches)
        return ends, (0.5, 0.75, 1.0), ends[-1]
    raise ValueError(f"Unsupported task_idx={task_idx}")


def _task0_grasp_frame(root: Path, ep_idx: int) -> int:
    for name in ("task0_radio_grasp_frames.json", "task0_radio_grasp_frames_v2.json"):
        path = root / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                entry = json.load(f).get("episodes", {}).get(str(ep_idx))
            if entry and bool(entry.get("grasp_detected", False)):
                return int(entry["grasp_frame"])
    raise FileNotFoundError(f"Missing task0 grasp metadata for episode {ep_idx}")


def _task0_passive_close_frame(root: Path, ep_idx: int, grasp_frame: int, press_end: int) -> int | None:
    grasp_path = root / "task0_radio_grasp_frames.json"
    if not grasp_path.exists():
        grasp_path = root / "task0_radio_grasp_frames_v2.json"
    with grasp_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    entry = payload.get("episodes", {}).get(str(ep_idx))
    if not entry:
        return None
    hold_hand = entry.get("hold_hand")
    state = np.asarray(pq.read_table(_data_path(root, 0, ep_idx), columns=["observation.state"]).column("observation.state").to_pylist(), dtype=np.float32)
    left_gripper = 2.0 * (state[:, 193:195].sum(axis=-1) / 0.1) - 1.0
    right_gripper = 2.0 * (state[:, 232:234].sum(axis=-1) / 0.1) - 1.0
    passive = left_gripper if hold_hand == "right" else right_gripper
    threshold = payload.get("meta", {}).get("grasp_quality_rule", {}).get("closed_threshold", -0.9)
    search = np.flatnonzero(passive[int(grasp_frame) : int(press_end)] <= float(threshold))
    if search.size == 0:
        return None
    frame = int(grasp_frame + int(search[0]))
    return frame if grasp_frame < frame < press_end else None


def _rft_value_spec(root: Path, task_idx: int, ep_idx: int, episode_len: int, bin_count: int):
    path = root / "reward_info" / "aligned" / f"task-{task_idx:04d}" / f"episode_{ep_idx:08d}.json"
    if not path.exists():
        path = root / "reward_info" / "aligned" / f"task-{task_idx:04d}" / f"episode_{ep_idx % 10000:08d}.json"
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    stable = np.full(10, -1, dtype=np.int32)
    progress_bin = 0
    for event in payload.get("events", []):
        aligned_step = int(event["aligned_step"])
        reward = float(event["reward"])
        delta = int(round(reward * float(bin_count - 1) / 10.0))
        if delta <= 0:
            continue
        for _ in range(delta):
            if progress_bin < 10:
                stable[progress_bin] = int(np.clip(aligned_step, 0, max(episode_len - 1, 0)))
            progress_bin += 1
    return stable, min(progress_bin, bin_count - 1), None, None


def _payload_from_boundaries(boundaries, anchors, mask_end):
    stable = np.full(10, -1, dtype=np.int32)
    for idx, end in enumerate(boundaries):
        stable[idx] = int(end)
    anchor_array = None if anchors is None else np.asarray(anchors, dtype=np.float32)
    return stable, len(boundaries), anchor_array, mask_end


def _stage_token(stable: np.ndarray, final_bin: int, bin_count: int, frame_idx: int, task_idx: int) -> int:
    num_subtasks = max(int(bin_count) - 1, 1)
    if final_bin <= 0:
        stage_idx = 0
    else:
        entries = [0] + [int(stable[i]) for i in range(final_bin)]
        stage_idx = final_bin
        for pos in range(1, final_bin + 1):
            if frame_idx < entries[pos]:
                stage_idx = pos - 1
                break
        stage_idx = min(stage_idx, max(num_subtasks - 1, 0))
    if task_idx == 22 and 4 <= stage_idx <= 7:
        stage_idx += 2
    return int(np.clip(stage_idx, 0, 9))


def _local_progress(stable: np.ndarray, final_bin: int, bin_count: int, frame_idx: int, episode_len: int) -> float:
    num_subtasks = max(int(bin_count) - 1, 1)
    if final_bin <= 0:
        return float(np.clip(frame_idx / max(float(episode_len), 1.0), 0.0, 1.0))
    entries = [0] + [int(stable[i]) for i in range(final_bin)]
    for stage_idx in range(1, final_bin + 1):
        start, end = entries[stage_idx - 1], entries[stage_idx]
        if frame_idx < end:
            return float(np.clip((frame_idx - start) / max(float(end - start), 1.0), 0.0, 1.0))
    if final_bin >= num_subtasks:
        return 1.0
    return float(np.clip((frame_idx - entries[final_bin]) / max(float(episode_len - entries[final_bin]), 1.0), 0.0, 1.0))


def _total_progress(stable: np.ndarray, final_bin: int, bin_count: int, frame_idx: int, episode_len: int, anchors):
    num_subtasks = max(int(bin_count) - 1, 1)
    if final_bin <= 0:
        return float(0.3 * frame_idx / max(float(episode_len), 1.0) / float(num_subtasks))
    entries = [0] + [int(stable[i]) for i in range(final_bin)]
    if anchors is not None:
        for stage_idx in range(1, final_bin + 1):
            start, end = entries[stage_idx - 1], entries[stage_idx]
            if frame_idx < end:
                frac = (frame_idx - start) / max(float(end - start), 1.0)
                p0 = 0.0 if stage_idx == 1 else float(anchors[stage_idx - 2])
                p1 = float(anchors[stage_idx - 1])
                return float(p0 + frac * (p1 - p0))
        return float(anchors[min(final_bin - 1, len(anchors) - 1)]) if final_bin >= num_subtasks else float(anchors[final_bin - 1])
    for stage_idx in range(1, final_bin + 1):
        start, end = entries[stage_idx - 1], entries[stage_idx]
        if frame_idx < end:
            frac = (frame_idx - start) / max(float(end - start), 1.0)
            return float((stage_idx - 1 + frac) / float(num_subtasks))
    return 1.0 if final_bin >= num_subtasks else float(final_bin / float(num_subtasks))


class B1KLocalTemporalDataset(Dataset):
    def __init__(
        self,
        *,
        source: B1KSourceConfig,
        camera_names: list[str],
        n_obs_steps: int,
        frame_gap: int,
        seed: int,
        chunk_streaming_using_keyframe: bool = False,
        temporal_all_windows_per_chunk: bool = False,
        eval_frame_gap: int | None = None,
    ):
        self.source = source
        self.root = Path(source.root).expanduser()
        self.task_idx = TASK_NAME_TO_INDEX[source.task_name]
        self.camera_names = list(camera_names)
        self.sequence_len = int(n_obs_steps) + 1
        self.frame_gap = int(frame_gap)
        self.rng = random.Random(seed)
        self.chunk_streaming_using_keyframe = bool(chunk_streaming_using_keyframe)
        self.eval_frame_gap = None if eval_frame_gap is None else int(eval_frame_gap)
        self.q_scores = _load_q_scores(self.root)
        self.lengths = _load_episode_lengths(self.root)
        self.episode_ids = _episode_ids_for_source(
            self.root,
            source.task_name,
            source.episodes,
            self.q_scores,
            source.use_q_score_1_only,
        )
        self.is_rft = bool(self.q_scores) or "rft" in str(self.root).lower()
        self.inputs = B1kInputs()
        self.cache: dict[int, tuple[np.ndarray, int, int, np.ndarray | None, int | None]] = {}
        self.windows = self._build_windows(all_windows=temporal_all_windows_per_chunk)

    def _build_windows(self, *, all_windows: bool):
        span = (self.sequence_len - 1) * self.frame_gap
        windows = []
        for ep_idx in self.episode_ids:
            ep_len = self.lengths[int(ep_idx)]
            if self.eval_frame_gap is not None:
                windows.extend((ep_idx, current) for current in range(0, ep_len, self.eval_frame_gap))
                continue
            max_start = ep_len - span - 1
            if max_start < 0:
                continue
            if all_windows:
                windows.extend((ep_idx, start) for start in range(max_start + 1))
            elif self.chunk_streaming_using_keyframe:
                chunk_size = 250
                for chunk_start in range(0, ep_len, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, ep_len)
                    if chunk_end - chunk_start > span:
                        windows.append((ep_idx, chunk_start))
            else:
                windows.extend((ep_idx, start) for start in range(0, max_start + 1, self.frame_gap))
        if not windows:
            raise ValueError(f"No valid temporal windows for {self.source}")
        return windows

    def __len__(self):
        return len(self.windows)

    def _value_payload(self, ep_idx: int):
        if ep_idx in self.cache:
            return self.cache[ep_idx]
        ep_len = self.lengths[int(ep_idx)]
        bin_count = TASK_PROGRESS_BIN_COUNTS[self.task_idx]
        if self.is_rft:
            stable, final_bin, anchors, mask_end = _rft_value_spec(self.root, self.task_idx, ep_idx, ep_len, bin_count)
        else:
            boundaries, anchors_tuple, mask_end = _expert_annotation_spec(self.root, self.task_idx, ep_idx, ep_len)
            stable, final_bin, anchors, mask_end = _payload_from_boundaries(boundaries, anchors_tuple, mask_end)
            if len(boundaries) + 1 != bin_count:
                raise ValueError(
                    f"Expert progress bins do not match task config: task_idx={self.task_idx}, "
                    f"boundaries={len(boundaries)}, expected_bin_count={bin_count}"
                )
        payload = (stable, final_bin, bin_count, anchors, mask_end)
        self.cache[ep_idx] = payload
        return payload

    def __getitem__(self, idx):
        ep_idx, chunk_start = self.windows[int(idx) % len(self.windows)]
        span = (self.sequence_len - 1) * self.frame_gap
        if self.eval_frame_gap is not None:
            current = int(chunk_start)
            frame_indices = [
                max(0, current - (self.sequence_len - 1 - i) * self.frame_gap)
                for i in range(self.sequence_len)
            ]
        elif self.source.use_for_validation or not self.chunk_streaming_using_keyframe:
            start = chunk_start
            frame_indices = [int(start + i * self.frame_gap) for i in range(self.sequence_len)]
        else:
            max_offset = max(0, min(250, self.lengths[ep_idx] - chunk_start) - span - 1)
            start = chunk_start + self.rng.randint(0, max_offset)
            frame_indices = [int(start + i * self.frame_gap) for i in range(self.sequence_len)]
        rows = _read_parquet_rows(_data_path(self.root, self.task_idx, ep_idx), frame_indices)

        raw = {
            "observation/state": rows["observation.state"],
            "task_index": rows["task_index"],
            "timestamp": rows["timestamp"],
            "episode_index": rows["episode_index"],
            "index": rows["index"],
        }
        for cam in ("observation.images.rgb.head", "observation.images.rgb.left_wrist", "observation.images.rgb.right_wrist"):
            raw[CAMERA_TO_B1K_KEY[cam]] = _decode_video_frames(
                _video_path(self.root, self.task_idx, ep_idx, cam),
                frame_indices,
            )

        stable, final_bin, bin_count, anchors, mask_end = self._value_payload(ep_idx)
        stage = []
        inner = []
        total = []
        progress_mask = []
        for frame_idx in frame_indices:
            stage.append(_stage_token(stable, final_bin, bin_count, frame_idx, self.task_idx))
            inner.append(_local_progress(stable, final_bin, bin_count, frame_idx, self.lengths[ep_idx]))
            total.append(_total_progress(stable, final_bin, bin_count, frame_idx, self.lengths[ep_idx], anchors))
            progress_mask.append(False if mask_end is not None and frame_idx >= int(mask_end) else True)
        raw["stage_token_index"] = np.asarray(stage, dtype=np.int32)
        raw["stage_sum"] = np.full((self.sequence_len,), max(int(bin_count) - 1, 1), dtype=np.int32)
        raw["progress_margin_target"] = np.asarray(inner, dtype=np.float32)
        raw["progress_target"] = np.asarray(total, dtype=np.float32)
        raw["progress_mask"] = np.asarray(progress_mask, dtype=np.bool_)
        raw["progress_loss_weight"] = np.ones((self.sequence_len,), dtype=np.float32)
        raw["ep_idx"] = np.full((self.sequence_len,), int(ep_idx), dtype=np.int64)
        raw["frame_idx"] = np.asarray(frame_indices, dtype=np.int64)
        raw["q_score"] = np.full((self.sequence_len,), float(self.q_scores.get(ep_idx, 1.0)), dtype=np.float32)
        raw["reward"] = np.zeros((self.sequence_len,), dtype=np.float32)

        transformed = self.inputs(raw)
        transformed["task"] = TASK_TEXT_BY_INDEX.get(self.task_idx, self.source.task_name)
        return transformed
