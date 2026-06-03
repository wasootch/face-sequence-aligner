"""
Video encoding: cross-fade between aligned frames and export to MP4.

Uses moviepy for encoding. Each aligned frame is held for `hold_frames` frames,
with a `transition_frames`-length cross-fade between consecutive images.
"""

import threading
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from aligner import AlignedFrame


def _crossfade(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Blend two BGR frames; t=0 → a, t=1 → b."""
    return cv2.addWeighted(a, 1.0 - t, b, t, 0).astype(np.uint8)


def build_frame_sequence(
    frames: list[AlignedFrame],
    fps: int = 30,
    hold_seconds: float = 1.0,
    transition_seconds: float = 1.0,
) -> list[np.ndarray]:
    """
    Build the full list of BGR frames for the video.

    Layout per pair (A, B):
      hold_frames of A  →  transition_frames cross-fade A→B
    The last image gets a trailing hold with no transition after it.
    """
    hold_frames = max(1, round(hold_seconds * fps))
    transition_frames = max(1, round(transition_seconds * fps))

    result: list[np.ndarray] = []
    images = [f.image for f in frames]

    for i, img in enumerate(images):
        for _ in range(hold_frames):
            result.append(img)

        if i < len(images) - 1:
            next_img = images[i + 1]
            for j in range(transition_frames):
                t = (j + 1) / (transition_frames + 1)
                result.append(_crossfade(img, next_img, t))

    return result


def export_mp4(
    frames: list[AlignedFrame],
    output_path: Path,
    fps: int = 30,
    hold_seconds: float = 1.0,
    transition_seconds: float = 1.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Encode aligned frames to an MP4 file.

    progress_callback(current, total) is called on the encoding thread.
    Raises on failure.
    """
    if not frames:
        raise ValueError("No frames to export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_seq = build_frame_sequence(frames, fps, hold_seconds, transition_seconds)
    total = len(frame_seq)

    h, w = frame_seq[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    try:
        for i, frame in enumerate(frame_seq):
            writer.write(frame)
            if progress_callback:
                progress_callback(i + 1, total)
    finally:
        writer.release()

    # Re-encode with H.264 via ffmpeg if available (better compatibility).
    # Falls back silently to the mp4v output already written.
    _try_reencode_h264(output_path)


def _try_reencode_h264(path: Path) -> None:
    """Attempt to re-encode the mp4v file to H.264 using ffmpeg."""
    import subprocess, shutil

    if not shutil.which("ffmpeg"):
        return

    tmp = path.with_suffix(".tmp.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(path),
                "-vcodec", "libx264",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(tmp),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        if result.returncode == 0:
            tmp.replace(path)
        else:
            tmp.unlink(missing_ok=True)
    except Exception:
        tmp.unlink(missing_ok=True)


def export_mp4_async(
    frames: list[AlignedFrame],
    output_path: Path,
    fps: int = 30,
    hold_seconds: float = 1.0,
    transition_seconds: float = 1.0,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    done_callback: Optional[Callable[[Optional[Exception]], None]] = None,
) -> threading.Thread:
    """Run export_mp4 on a background thread; returns the Thread."""

    def _run():
        try:
            export_mp4(
                frames, output_path, fps, hold_seconds,
                transition_seconds, progress_callback,
            )
            if done_callback:
                done_callback(None)
        except Exception as exc:
            if done_callback:
                done_callback(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
