from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from lerobot.common.datasets.b1k_sarm_dataset import collate_b1k_sarm
from models.clip_encoder import FrozenCLIPEncoder
from models.rewind_reward_model import RewardTransformer
from utils.train_utils import save_ckpt
from workspace.b1k_common import (
    B1KBaseWorkspace,
    encode_batch,
    make_b1k_datasets,
    make_loaders,
    make_scheduler,
    masked_mse,
    save_behavior_value_jsons,
    wandb_log,
    wandb_run,
)


class B1KReWiNDWorkspace(B1KBaseWorkspace):
    def build_model(self):
        cfg = self.cfg
        return RewardTransformer(
            d_model=cfg.model.d_model,
            vis_emb_dim=512,
            text_emb_dim=512,
            state_dim=cfg.model.state_dim,
            n_layers=cfg.model.n_layers,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
            num_cameras=len(self.camera_names),
        ).to(self.device)

    def train(self):
        cfg = self.cfg
        self.save_config()
        train_loader, val_loader = make_loaders(cfg)
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        reward_model = self.build_model()

        if cfg.model.resume_training:
            reward_model.load_state_dict(torch.load(Path(cfg.model.model_path), map_location=self.device)["model"])

        optimizer = torch.optim.AdamW(
            reward_model.parameters(),
            lr=cfg.optim.lr,
            betas=tuple(cfg.optim.betas),
            eps=cfg.optim.eps,
            weight_decay=cfg.optim.weight_decay,
        )
        scheduler = make_scheduler(optimizer, cfg)

        def step_batch(batch, training: bool):
            image_emb, lang_emb, state, lengths = encode_batch(
                batch, cfg, self.camera_names, self.device, clip_encoder
            )
            target = batch["total_progress"].to(self.device)
            pred = reward_model(image_emb, lang_emb, state, lengths)
            loss = masked_mse(pred, target, lengths)
            grad = 0.0
            if training:
                optimizer.zero_grad()
                loss.backward()
                grad = nn.utils.clip_grad_norm_(reward_model.parameters(), float("inf")).item()
                nn.utils.clip_grad_norm_(reward_model.parameters(), cfg.train.grad_clip)
                optimizer.step()
                scheduler.step()
            return {"loss": float(loss.item()), "grad_norm": grad}

        best_val = float("inf")
        step = 0
        with wandb_run(cfg):
            for epoch in range(1, cfg.train.num_epochs + 1):
                reward_model.train()
                with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
                    for batch in pbar:
                        metrics = step_batch(batch, training=True)
                        if step % cfg.train.log_every == 0:
                            wandb_log(
                                cfg,
                                {
                                    "train/total_loss": metrics["loss"],
                                    "train/lr": scheduler.get_last_lr()[0],
                                    "train/reward_grad_norm": metrics["grad_norm"],
                                    "epoch": epoch,
                                },
                                step,
                            )
                        pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
                        if step % cfg.train.save_every == 0:
                            save_ckpt(reward_model, optimizer, epoch, self.save_dir, f"reward_step_{step:06d}")
                        step += 1

                if epoch % cfg.train.eval_every == 0:
                    reward_model.eval()
                    total, count = 0.0, 0
                    with torch.no_grad():
                        for batch in tqdm(val_loader, desc="Validation"):
                            metrics = step_batch(batch, training=False)
                            total += metrics["loss"]
                            count += 1
                    val_loss = total / max(count, 1)
                    print(f"[Eval] Epoch {epoch} val_loss={val_loss:.6f}")
                    wandb_log(cfg, {"val/loss": val_loss}, step)
                    if val_loss < best_val:
                        best_val = val_loss
                        save_ckpt(reward_model, optimizer, epoch, self.save_dir, "reward_best")

                save_ckpt(reward_model, optimizer, epoch, self.save_dir, "reward_latest")

        save_ckpt(reward_model, optimizer, cfg.train.num_epochs, self.save_dir, "reward_final")
        print(f"Training done. Best val_loss = {best_val:.6f}")

    def eval(self):
        cfg = self.cfg
        dataset = make_b1k_datasets(cfg, for_eval=True)
        loader = torch.utils.data.DataLoader(dataset, collate_fn=collate_b1k_sarm, **cfg.eval_dataloader)
        clip_encoder = FrozenCLIPEncoder(cfg.encoders.vision_ckpt, self.device)
        reward_model = self.build_model()
        ckpt_path = Path(cfg.eval.ckpt_path) / cfg.eval.reward_model_name
        reward_model.load_state_dict(torch.load(ckpt_path, map_location=self.device)["model"])
        reward_model.eval()

        expected_series: dict[str, dict[str, float]] = {}
        raw_series: dict[str, dict[str, float]] = {}
        last_idx = cfg.model.n_obs_steps
        with torch.no_grad():
            for batch in tqdm(loader, desc="Export"):
                image_emb, lang_emb, state, lengths = encode_batch(
                    batch, cfg, self.camera_names, self.device, clip_encoder
                )
                pred = torch.clamp(reward_model(image_emb, lang_emb, state, lengths), 0.0, 1.0)
                for row in range(pred.shape[0]):
                    ep = str(int(batch["ep_idx"][row, last_idx].item()))
                    frame = str(int(batch["frame_idx"][row, last_idx].item()))
                    value = float(pred[row, last_idx].item())
                    expected_series.setdefault(ep, {})[frame] = value
                    raw_series.setdefault(ep, {})[frame] = value

        output_dir = Path(cfg.eval.output_dir or (self.save_dir / "b1k_json"))
        save_behavior_value_jsons(output_dir, cfg.eval.output_suffix, expected_series, raw_series, cfg)
        print(f"[Export] Wrote behavior-style JSON files to: {output_dir}")

    def eval_raw_data(self):
        raise NotImplementedError("B1K migration exports validation JSON via eval(); raw robot trajectory eval is unchanged.")
