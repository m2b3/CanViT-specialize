"""ADE20K probe training (CanViT canvas probes + DINOv3 baseline probes).

Usage:
    uv run python -m canvit_probes.training.ade20k train ...
    uv run python -m canvit_probes.training.ade20k train-dinov3-probe ...
"""

from typing import Annotated

import tyro

from canvit_probes.training.ade20k.train_dinov3 import DINOv3ProbeTrainConfig
from canvit_probes.training.ade20k.train_dinov3 import train as run_train_dinov3
from canvit_probes.training.ade20k.train_canvit import train as run_train
from canvit_probes.training.ade20k.config import Config as TrainConfig


def main() -> None:
    cmd = tyro.cli(
        Annotated[TrainConfig, tyro.conf.subcommand("train")]
        | Annotated[DINOv3ProbeTrainConfig, tyro.conf.subcommand("train-dinov3-probe")]
    )
    match cmd:
        case TrainConfig():
            run_train(cmd)
        case DINOv3ProbeTrainConfig():
            run_train_dinov3(cmd)


if __name__ == "__main__":
    main()
