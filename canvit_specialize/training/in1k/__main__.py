"""DINOv3 IN1K linear probe: extraction and training.

Usage:
    uv run python -m canvit_specialize.training.in1k extract ...
    uv run python -m canvit_specialize.training.in1k train ...
"""

from typing import Annotated, Union

import tyro

from canvit_specialize.training.in1k.config import ExtractionConfig, TrainConfig
from canvit_specialize.training.in1k.extract import run_extraction
from canvit_specialize.training.in1k.train import run_training

_Command = Union[
    Annotated[ExtractionConfig, tyro.conf.subcommand("extract")],
    Annotated[TrainConfig, tyro.conf.subcommand("train")],
]


def main() -> None:
    cmd: ExtractionConfig | TrainConfig = tyro.cli(_Command)  # pyright: ignore[reportCallIssue,reportArgumentType]
    match cmd:
        case ExtractionConfig():
            run_extraction(cmd)
        case TrainConfig():
            run_training(cmd)


if __name__ == "__main__":
    main()
