import json
from pathlib import Path
from typing import Any, Dict, List

from .config import PipelineConfig


class DatasetScanner:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def get_video_items(self) -> List[Dict[str, str]]:
        video_paths = sorted(Path(self.config.videos_dir).glob("*.mp4"))
        items: List[Dict[str, str]] = []

        for video_path in video_paths:
            video_name = video_path.stem
            shots_path = Path(self.config.shots_dir) / f"{video_name}.json"
            output_path = Path(self.config.output_dir) / f"{video_name}.pkl"

            if not shots_path.exists():
                print(f"[WARN] Missing shots file for {video_name}: {shots_path}")
                continue

            items.append(
                {
                    "video_name": video_name,
                    "video_path": str(video_path),
                    "shots_path": str(shots_path),
                    "output_path": str(output_path),
                }
            )

        if self.config.video_ids:
            items_by_video_name = {item["video_name"]: item for item in items}
            filtered_items = []
            missing_video_ids = []

            for video_id in self.config.video_ids:
                normalized_video_id = str(video_id)
                item = items_by_video_name.get(normalized_video_id)
                if item is None:
                    missing_video_ids.append(normalized_video_id)
                    continue
                filtered_items.append(item)

            for video_id in missing_video_ids:
                print(f"[WARN] Missing video or shots file for requested video id: {video_id}")

            return filtered_items

        start = max(0, int(self.config.start_index))
        end = self.config.end_index

        if start >= len(items):
            return []
        if end is None:
            return items[start:]
        return items[start:end + 1]


class ShotLoader:
    REQUIRED_FIELDS = [
        "shot_id",
        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
    ]

    def load(self, shots_path: str) -> List[Dict[str, Any]]:
        with open(shots_path, "r", encoding="utf-8") as file:
            shots = json.load(file)

        if not isinstance(shots, list):
            raise ValueError(f"Shots JSON must be a list: {shots_path}")

        for shot in shots:
            for field in self.REQUIRED_FIELDS:
                if field not in shot:
                    raise ValueError(f"Missing field {field} in shot: {shot}")
            if int(shot["start_frame"]) > int(shot["end_frame"]):
                raise ValueError(f"Invalid shot frame range: {shot}")

        shots = sorted(
            shots,
            key=lambda item: (int(item["start_frame"]), int(item["shot_id"])),
        )
        return shots
