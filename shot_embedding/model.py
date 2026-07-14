from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from .config import PipelineConfig


class CLIPEmbedder:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = torch.device(
            config.device if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        )
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        import open_clip

        pretrained = self.config.clip_pretrained
        pretrained_value = str(pretrained)
        if Path(pretrained_value).exists():
            pretrained_value = pretrained_value

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.config.clip_model_name,
            pretrained=pretrained_value,
            device=self.device,
        )
        self.model.eval()

    @staticmethod
    def _l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norm, eps, None)

    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if len(images) == 0:
            return np.empty((0, 0), dtype=np.float32)

        all_embeddings: List[np.ndarray] = []

        with torch.no_grad():
            for start in range(0, len(images), self.config.batch_size):
                batch_images = images[start:start + self.config.batch_size]
                batch_tensor = torch.stack(
                    [self.preprocess(image) for image in batch_images]
                ).to(self.device)

                embeddings = self.model.encode_image(batch_tensor)
                embeddings = embeddings.float()
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                all_embeddings.append(embeddings.cpu().numpy())

        embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        embeddings = self._l2_normalize_np(embeddings).astype(np.float32)
        return embeddings
