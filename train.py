# torchrun --nproc_per_node=NUMER_OF_GPU train.py (recommended for training)
# But you don't multi-GPU then you can run python3 train.py

import hydra
from omegaconf import DictConfig
from ycsai import Engine

@hydra.main(config_path="ycsai/cfg", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    model = Engine(cfg)

if __name__ == "__main__":
    main()
