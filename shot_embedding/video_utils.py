from pathlib import Path
from typing import Optional, Tuple

import cv2
from PIL import Image


class VideoFrameReader:
    def __init__(self, video_path: str):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next_frame_index: Optional[int] = None

    def _read_frame_once(self, frame_index: int) -> Optional[Image.Image]:
        if self.frame_count <= 0:
            return None

        frame_index = int(max(0, min(frame_index, self.frame_count - 1)))

        if self._next_frame_index is None or frame_index < self._next_frame_index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            self._next_frame_index = frame_index
        elif frame_index > self._next_frame_index:
            steps_to_skip = frame_index - self._next_frame_index
            for _ in range(steps_to_skip):
                if not self.cap.grab():
                    return None
            self._next_frame_index = frame_index

        ok, frame_bgr = self.cap.read()
        if not ok or frame_bgr is None:
            return None

        self._next_frame_index = frame_index + 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def read_frame(self, frame_index: int) -> Optional[Image.Image]:
        return self._read_frame_once(frame_index)

    def read_frame_with_fallback(
        self,
        frame_index: int,
        fallback_radius: int = 6,
    ) -> Tuple[Optional[Image.Image], Optional[int]]:
        image = self._read_frame_once(frame_index)
        if image is not None:
            return image, int(frame_index)

        for offset in range(1, max(0, int(fallback_radius)) + 1):
            candidate_indices = [frame_index - offset, frame_index + offset]
            for candidate_index in candidate_indices:
                if candidate_index < 0 or candidate_index >= self.frame_count:
                    continue
                image = self._read_frame_once(candidate_index)
                if image is not None:
                    return image, int(candidate_index)

        return None, None

    def close(self):
        self.cap.release()
