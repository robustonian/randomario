"""Video recording for replays.

Pixel-art-friendly quality: frames are upscaled with nearest-neighbor
(crisp pixels, no smearing) and encoded near-losslessly with libx264
(crf 10, slow preset) through an ffmpeg pipe. Falls back to OpenCV's
mp4v writer when ffmpeg is unavailable.
"""
import os
import shutil
import subprocess
from typing import Optional

import cv2
import numpy as np

SRC_W, SRC_H = 256, 240


class VideoRecorder:
    def __init__(self, path: str, scale: int = 4, fps: int = 60, size=None):
        """size: explicit output (w, h); default = game frame x scale.
        Frames passed to add() are nearest-neighbor resized to fit."""
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.path = path
        self.scale = scale
        self.fps = fps
        self.size = size if size is not None else (SRC_W * scale, SRC_H * scale)
        self.frames = 0
        self._proc: Optional[subprocess.Popen] = None
        self._cv: Optional[cv2.VideoWriter] = None

        if shutil.which('ffmpeg'):
            self._proc = subprocess.Popen(
                ['ffmpeg', '-y', '-loglevel', 'error',
                 '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                 '-s', f'{self.size[0]}x{self.size[1]}', '-r', str(fps), '-i', '-',
                 '-c:v', 'libx264', '-preset', 'slow', '-crf', '10',
                 '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path],
                stdin=subprocess.PIPE)
        else:
            self._cv = cv2.VideoWriter(
                path, cv2.VideoWriter_fourcc(*'mp4v'), fps, self.size)

    def add(self, frame: np.ndarray, repeat: int = 1):
        """frame: RGB uint8 (any size). repeat holds it for N frames."""
        frame = np.ascontiguousarray(frame)
        if (frame.shape[1], frame.shape[0]) != self.size:
            big = cv2.resize(frame, self.size, interpolation=cv2.INTER_NEAREST)
        else:
            big = frame
        if self._proc is not None:
            data = big.tobytes()
            for _ in range(repeat):
                self._proc.stdin.write(data)
        else:
            bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
            for _ in range(repeat):
                self._cv.write(bgr)
        self.frames += repeat

    def close(self):
        if self._proc is not None:
            try:
                self._proc.stdin.close()
                self._proc.wait(timeout=120)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._cv is not None:
            self._cv.release()
            self._cv = None
