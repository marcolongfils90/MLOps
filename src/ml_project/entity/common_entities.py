"""Module containing common entities used by the pipelines."""
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class BaseModelConfig:
    root_dir: Path
    model_path: Path
    untrained_model_path: Path
    params_input_size: list
    params_include_top: bool
    params_num_classes: int
    params_weights: str
    params_learning_rate: float
