import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

@hydra.main(config_path="config", config_name=None, version_base="1.1")
def main(cfg: DictConfig):
    workspace = instantiate(cfg)
    mode = OmegaConf.select(cfg, "cfg.general.mode", default="train")
    if mode == "train":
        workspace.train()
    elif mode in {"eval", "export"}:
        workspace.eval()
    else:
        raise ValueError(f"Unsupported mode={mode!r}. Use 'train' or 'eval'.")

if __name__ == "__main__":
    main()
