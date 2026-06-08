from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.clip_encoder import FrozenCLIPEncoder
from models.stage_estimator import StageTransformer
from models.subtask_estimator import SubtaskTransformer
from utils.train_utils import save_ckpt
from workspace.b1k_common import (
    B1KBaseWorkspace,
    encode_batch,
    make_b1k_datasets,
    make_loaders,
    make_scheduler,
    masked_cross_entropy,
    masked_mse,
    save_behavior_value_jsons,
    wandb_log,
    wandb_run,
)
from lerobot.common.datasets.b1k_sarm_dataset import collate_b1k_sarm


class B1KSARMWorkspace(B1KBaseWorkspace):
    def gen_stage_emb(self, num_classes, targets):
        stage_idx = targets.long().clamp(min=0, max=num_classes - 1)
        return torch.eye(num_classes, device=targets.device)[stage_idx].unsqueeze(1)

    def build_models(self):
        cfg = self.cfg
        stage_model = StageTransformer(
            d_model=cfg.model.d_model,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=cfg.model.state_dim,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
            num_cameras=len(self.camera_names),
            num_classes_sparse=cfg.model.num_classes,
            num_classes_dense=cfg.model.num_classes,
        ).to(self.device)
        subtask_model = SubtaskTransformer(
            d_model=cfg.model.d_model,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=cfg.model.state_dim,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
            num_cameras=len(self.camera_names),
        ).to(self.device)
        return stage_model, subtask_model

    def train(self):
        cfg = self.cfg
        self.save_config()
        train_loader, val_loader = make_loaders(cfg)
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        stage_model, subtask_model = self.build_models()

        if cfg.model.resume_training:
            ckpt_dir = Path(cfg.model.model_path)
            stage_model.load_state_dict(torch.load(ckpt_dir / "stage_latest.pt", map_location=self.device)["model"])
            subtask_model.load_state_dict(torch.load(ckpt_dir / "subtask_latest.pt", map_location=self.device)["model"])

        stage_optimizer = torch.optim.AdamW(
            stage_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        subtask_optimizer = torch.optim.AdamW(
            subtask_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        stage_scheduler = make_scheduler(stage_optimizer, cfg)
        subtask_scheduler = make_scheduler(subtask_optimizer, cfg)

        def step_batch(batch, training: bool):
            image_emb, lang_emb, state, lengths = encode_batch(
                batch, cfg, self.camera_names, self.device, clip_encoder
            )
            targets = batch["targets"].to(self.device)
            gt_stage = torch.floor(targets).long().clamp(min=0, max=cfg.model.num_classes - 1)
            gt_inner = torch.remainder(targets, 1.0)

            stage_logits = stage_model(image_emb, lang_emb, state, lengths, scheme="sparse")
            if training and torch.rand(1).item() < 0.5:
                stage_emb = self.gen_stage_emb(cfg.model.num_classes, targets)
            else:
                stage_idx = stage_logits.argmax(dim=-1)
                stage_emb = F.one_hot(stage_idx, num_classes=cfg.model.num_classes).float().unsqueeze(1)
            subtask_pred = subtask_model(image_emb, lang_emb, state, lengths, stage_emb, scheme="sparse")

            stage_loss = masked_cross_entropy(stage_logits, gt_stage, lengths)
            subtask_loss = masked_mse(subtask_pred, gt_inner, lengths)
            total_loss = stage_loss + subtask_loss

            if training:
                subtask_optimizer.zero_grad()
                subtask_loss.backward(retain_graph=True)
                subtask_grad = nn.utils.clip_grad_norm_(subtask_model.parameters(), float("inf")).item()
                nn.utils.clip_grad_norm_(subtask_model.parameters(), cfg.train.grad_clip)
                subtask_optimizer.step()
                subtask_scheduler.step()

                stage_optimizer.zero_grad()
                stage_loss.backward()
                stage_grad = nn.utils.clip_grad_norm_(stage_model.parameters(), float("inf")).item()
                nn.utils.clip_grad_norm_(stage_model.parameters(), cfg.train.grad_clip)
                stage_optimizer.step()
                stage_scheduler.step()
            else:
                stage_grad = 0.0
                subtask_grad = 0.0

            return {
                "stage_loss": float(stage_loss.item()),
                "subtask_loss": float(subtask_loss.item()),
                "total_loss": float(total_loss.item()),
                "stage_grad_norm": stage_grad,
                "subtask_grad_norm": subtask_grad,
            }

        best_val = float("inf")
        step = 0
        with wandb_run(cfg):
            for epoch in range(1, cfg.train.num_epochs + 1):
                stage_model.train()
                subtask_model.train()
                with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
                    for batch in pbar:
                        metrics = step_batch(batch, training=True)
                        if step % cfg.train.log_every == 0:
                            wandb_log(
                                cfg,
                                {
                                    "train/stage_loss": metrics["stage_loss"],
                                    "train/subtask_loss": metrics["subtask_loss"],
                                    "train/total_loss": metrics["total_loss"],
                                    "train/lr": subtask_scheduler.get_last_lr()[0],
                                    "train/stage_grad_norm": metrics["stage_grad_norm"],
                                    "train/subtask_grad_norm": metrics["subtask_grad_norm"],
                                    "epoch": epoch,
                                },
                                step,
                            )
                        pbar.set_postfix(loss=f"{metrics['total_loss']:.4f}")
                        if step % cfg.train.save_every == 0:
                            save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, f"stage_step_{step:06d}")
                            save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, f"subtask_step_{step:06d}")
                        step += 1

                if epoch % cfg.train.eval_every == 0:
                    stage_model.eval()
                    subtask_model.eval()
                    total, count = 0.0, 0
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc="Validation"):
                            metrics = step_batch(batch, training=False)
                            total += metrics["total_loss"]
                            count += 1
                    val_loss = total / max(count, 1)
                    print(f"[Eval] Epoch {epoch} val_loss={val_loss:.6f}")
                    wandb_log(cfg, {"val/loss": val_loss}, step)
                    if val_loss < best_val:
                        best_val = val_loss
                        save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, "stage_best")
                        save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, "subtask_best")

                save_ckpt(stage_model, stage_optimizer, epoch, self.save_dir, "stage_latest")
                save_ckpt(subtask_model, subtask_optimizer, epoch, self.save_dir, "subtask_latest")

        save_ckpt(stage_model, stage_optimizer, cfg.train.num_epochs, self.save_dir, "stage_final")
        save_ckpt(subtask_model, subtask_optimizer, cfg.train.num_epochs, self.save_dir, "subtask_final")
        print(f"Training done. Best val_loss = {best_val:.6f}")

    def eval(self):
        cfg = self.cfg
        dataset = make_b1k_datasets(cfg, for_eval=True)
        loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_b1k_sarm, **cfg.eval_dataloader)
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        stage_model, subtask_model = self.build_models()

        ckpt_dir = Path(cfg.eval.ckpt_path)
        stage_model.load_state_dict(torch.load(ckpt_dir / cfg.eval.stage_model, map_location=self.device)["model"])
        subtask_model.load_state_dict(torch.load(ckpt_dir / cfg.eval.subtask_model, map_location=self.device)["model"])
        stage_model.eval()
        subtask_model.eval()

        expected_series: dict[str, dict[str, float]] = {}
        raw_series: dict[str, dict[str, float]] = {}
        denominator = float(cfg.eval.progress_denominator)
        last_idx = cfg.model.n_obs_steps

        with torch.no_grad():
            for batch in tqdm(loader, desc="Export"):
                image_emb, lang_emb, state, lengths = encode_batch(
                    batch, cfg, self.camera_names, self.device, clip_encoder
                )
                stage_logits = stage_model(image_emb, lang_emb, state, lengths, scheme="sparse")
                stage_idx = stage_logits.argmax(dim=-1)
                stage_emb = F.one_hot(stage_idx, num_classes=cfg.model.num_classes).float().unsqueeze(1)
                subtask_pred = subtask_model(image_emb, lang_emb, state, lengths, stage_emb, scheme="sparse")
                expected = torch.clamp((stage_idx.float() + subtask_pred) / denominator, 0.0, 1.0)

                for row in range(expected.shape[0]):
                    ep = str(int(batch["ep_idx"][row, last_idx].item()))
                    frame = str(int(batch["frame_idx"][row, last_idx].item()))
                    expected_series.setdefault(ep, {})[frame] = float(expected[row, last_idx].item())
                    raw_series.setdefault(ep, {})[frame] = float(subtask_pred[row, last_idx].item())

        output_dir = Path(cfg.eval.output_dir or (self.save_dir / "b1k_json"))
        save_behavior_value_jsons(output_dir, cfg.eval.output_suffix, expected_series, raw_series, cfg)
        print(f"[Export] Wrote behavior-style JSON files to: {output_dir}")

    def eval_raw_data(self):
        raise NotImplementedError("B1K migration exports validation JSON via eval(); raw robot trajectory eval is unchanged.")
