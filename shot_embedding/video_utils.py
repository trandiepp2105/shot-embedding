from pathlib import Path
from typing import Optional

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

    def read_frame(self, frame_index: int) -> Optional[Image.Image]:
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

    def close(self):
        self.cap.release()
