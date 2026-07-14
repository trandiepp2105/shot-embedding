from dataclasses import dataclass
from typing import Optional


@dataclass
class PipelineConfig:
    videos_dir: str
    shots_dir: str
    output_dir: str

    frame_step: int = 5
    similarity_threshold: float = 0.90
    pool_temperature: float = 10.0

    start_index: int = 0
    end_index: Optional[int] = None

    clip_model_name: str = "ViT-H-14-quickgelu"
    clip_pretrained: str = "dfn5b"

    batch_size: int = 32
    device: str = "cuda"

    save_dtype: str = "float16"
    overwrite: bool = False
