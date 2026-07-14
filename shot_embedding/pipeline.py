import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from .config import PipelineConfig
from .io_utils import DatasetScanner, ShotLoader
from .model import CLIPEmbedder
from .video_utils import VideoFrameReader


class ShotEmbeddingBuilder:
    def __init__(self, config: PipelineConfig, embedder: CLIPEmbedder):
        self.config = config
        self.embedder = embedder

    @staticmethod
    def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.clip(norms, eps, None)

    @staticmethod
    def _l2_normalize_vector(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        norm = float(np.linalg.norm(x))
        return x / max(norm, eps)

    def sample_shot_frames(
        self,
        reader: VideoFrameReader,
        shot: Dict[str, Any],
    ) -> Dict[str, Any]:
        start_frame = int(shot["start_frame"])
        end_frame = int(shot["end_frame"])

        sampled_indices = list(range(start_frame, end_frame + 1, self.config.frame_step))
        if len(sampled_indices) == 0:
            sampled_indices = [(start_frame + end_frame) // 2]

        valid_indices = [
            frame_idx
            for frame_idx in sampled_indices
            if 0 <= frame_idx < reader.frame_count
        ]

        if len(valid_indices) == 0:
            midpoint = (start_frame + end_frame) // 2
            midpoint = int(max(0, min(midpoint, reader.frame_count - 1)))
            valid_indices = [midpoint]

        return {
            "shot": shot,
            "sampled_frame_indices": valid_indices,
        }

    def _select_keyframe_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        selected_local_indices = [0]
        last_key_embedding = embeddings[0]

        for index in range(1, len(embeddings)):
            similarity = float(np.dot(last_key_embedding, embeddings[index]))
            if similarity < self.config.similarity_threshold:
                selected_local_indices.append(index)
                last_key_embedding = embeddings[index]

        return embeddings[selected_local_indices].astype(np.float32)

    def _pool_shot_embedding(self, keyframe_embeddings: np.ndarray) -> np.ndarray:
        normalized = self._l2_normalize_rows(keyframe_embeddings.astype(np.float32))
        center = self._l2_normalize_vector(normalized.mean(axis=0))
        similarities = normalized @ center

        logits = similarities * self.config.pool_temperature
        logits = logits - np.max(logits)
        weights = np.exp(logits)
        weights = weights / np.clip(weights.sum(), 1e-12, None)

        pooled = (weights[:, None] * normalized).sum(axis=0)
        pooled = self._l2_normalize_vector(pooled).astype(np.float32)
        return pooled

    def build_shot_output(
        self,
        shot: Dict[str, Any],
        sampled_frame_indices: List[int],
        embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        if embeddings.shape[0] != len(sampled_frame_indices):
            raise ValueError(
                f"Embedding count mismatch for shot_id={shot['shot_id']}: "
                f"{embeddings.shape[0]} vs {len(sampled_frame_indices)}"
            )

        keyframe_embeddings = self._select_keyframe_embeddings(embeddings)
        shot_embedding = self._pool_shot_embedding(keyframe_embeddings)

        if self.config.save_dtype == "float16":
            shot_embedding = shot_embedding.astype(np.float16)
        elif self.config.save_dtype == "float32":
            shot_embedding = shot_embedding.astype(np.float32)
        else:
            raise ValueError(f"Unsupported save_dtype: {self.config.save_dtype}")

        return {
            "shot_id": int(shot["shot_id"]),
            "embedding": shot_embedding,
            "start_time_sec": float(shot["start_time_sec"]),
            "end_time_sec": float(shot["end_time_sec"]),
            "start_frame": int(shot["start_frame"]),
            "end_frame": int(shot["end_frame"]),
            "duration_sec": float(shot["duration_sec"]),
        }


class VideoProcessor:
    def __init__(
        self,
        config: PipelineConfig,
        shot_loader: ShotLoader,
        embedding_builder: ShotEmbeddingBuilder,
    ):
        self.config = config
        self.shot_loader = shot_loader
        self.embedding_builder = embedding_builder

    def _encode_sampled_shots(
        self,
        reader: VideoFrameReader,
        sampled_shots: List[Dict[str, Any]],
        video_name: str,
    ) -> List[np.ndarray]:
        frame_refs: List[tuple[int, int]] = []
        embeddings_by_shot: List[List[np.ndarray]] = [list() for _ in sampled_shots]

        for shot_index, sampled_shot in enumerate(sampled_shots):
            for frame_index in sampled_shot["sampled_frame_indices"]:
                frame_refs.append((shot_index, frame_index))

        batch_starts = range(0, len(frame_refs), self.config.batch_size)
        for start in tqdm(
            batch_starts,
            total=(len(frame_refs) + self.config.batch_size - 1) // self.config.batch_size,
            desc=f"Encoding: {video_name}",
            leave=False,
        ):
            batch_refs = frame_refs[start:start + self.config.batch_size]
            batch_images: List[Image.Image] = []
            valid_batch_refs: List[int] = []

            for shot_index, frame_index in batch_refs:
                image = reader.read_frame(frame_index)
                if image is not None:
                    batch_images.append(image)
                    valid_batch_refs.append(shot_index)

            if not batch_images:
                continue

            batch_embeddings = self.embedding_builder.embedder.encode_images(batch_images)
            for shot_index, embedding in zip(valid_batch_refs, batch_embeddings):
                embeddings_by_shot[shot_index].append(embedding)

        outputs: List[np.ndarray] = []
        for shot_index, shot_embeddings in enumerate(embeddings_by_shot):
            if len(shot_embeddings) == 0:
                shot_id = sampled_shots[shot_index]["shot"]["shot_id"]
                raise RuntimeError(f"No embeddings collected for shot_id={shot_id}")
            outputs.append(np.stack(shot_embeddings, axis=0).astype(np.float32))

        return outputs

    def process_video(self, item: Dict[str, str]) -> Dict[str, Any]:
        video_name = item["video_name"]
        video_path = item["video_path"]
        shots_path = item["shots_path"]
        output_path = item["output_path"]

        shots = self.shot_loader.load(shots_path)
        reader = VideoFrameReader(video_path)
        sampled_shots: List[Dict[str, Any]] = []
        processed_shots: List[Dict[str, Any]] = []

        try:
            for shot in tqdm(shots, desc=f"Shots: {video_name}", leave=False):
                sampled_shots.append(self.embedding_builder.sample_shot_frames(reader, shot))

            embeddings_by_shot = self._encode_sampled_shots(reader, sampled_shots, video_name)

            for sampled_shot, embeddings in zip(sampled_shots, embeddings_by_shot):
                processed_shot = self.embedding_builder.build_shot_output(
                    shot=sampled_shot["shot"],
                    sampled_frame_indices=sampled_shot["sampled_frame_indices"],
                    embeddings=embeddings,
                )
                processed_shots.append(processed_shot)
        finally:
            reader.close()

        total_sampled_frames = sum(len(item["sampled_frame_indices"]) for item in sampled_shots)

        embedding_dim = 0
        for shot in processed_shots:
            embedding = shot["embedding"]
            if hasattr(embedding, "shape") and len(embedding.shape) == 1:
                embedding_dim = int(embedding.shape[0])
                break

        data = {
            "video_name": video_name,
            "video_path": video_path,
            "shots_path": shots_path,
            "fps": reader.fps,
            "frame_count": reader.frame_count,
            "width": reader.width,
            "height": reader.height,
            "frame_step": self.config.frame_step,
            "similarity_threshold": self.config.similarity_threshold,
            "shot_embedding_pooling": "softmax_centrality",
            "pool_temperature": self.config.pool_temperature,
            "clip_model_name": self.config.clip_model_name,
            "clip_pretrained": self.config.clip_pretrained,
            "embedding_dim": embedding_dim,
            "embedding_dtype": self.config.save_dtype,
            "num_shots": len(processed_shots),
            "total_sampled_frames": total_sampled_frames,
            "shots": processed_shots,
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as file:
            pickle.dump(data, file)

        return data


class BatchProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

        self.scanner = DatasetScanner(config)
        self.shot_loader = ShotLoader()
        self.embedder = CLIPEmbedder(config)
        self.embedding_builder = ShotEmbeddingBuilder(config, self.embedder)
        self.video_processor = VideoProcessor(
            config=config,
            shot_loader=self.shot_loader,
            embedding_builder=self.embedding_builder,
        )

    def run(self) -> Dict[str, Any]:
        items = self.scanner.get_video_items()
        print(f"Found {len(items)} videos to process")

        summary: Dict[str, Any] = {
            "done": [],
            "skipped": [],
            "failed": [],
        }

        for item in tqdm(items, desc="Processing videos"):
            output_path = Path(item["output_path"])

            try:
                if output_path.exists() and not self.config.overwrite:
                    print(f"[SKIP] {item['video_name']}")
                    summary["skipped"].append(item["video_name"])
                    continue

                output = self.video_processor.process_video(item)
                print(
                    f"[DONE] {item['video_name']} | "
                    f"num_shots={output['num_shots']} | "
                    f"embedding_dim={output['embedding_dim']}"
                )
                summary["done"].append(
                    {
                        "video_name": item["video_name"],
                        "output_path": item["output_path"],
                        "num_shots": output["num_shots"],
                        "embedding_dim": output["embedding_dim"],
                    }
                )
            except Exception as error:
                print(f"[FAILED] {item['video_name']}: {error}")
                summary["failed"].append(
                    {
                        "video_name": item["video_name"],
                        "output_path": item["output_path"],
                        "error": repr(error),
                    }
                )

        return summary
