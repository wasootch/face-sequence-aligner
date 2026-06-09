"""
Video encoding: cross-fade between aligned frames and export to MP4.

Each aligned frame is held for `hold_frames` frames, with a cross-fade
transition between consecutive images.  An optional audio track is looped
(or trimmed) to match the video duration and mixed in during the H.264
re-encode step.  Audio requires ffmpeg on PATH; video-only export always works.
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
    hold_frames       = max(1, round(hold_seconds       * fps))
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
    audio_path: Optional[Path] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> None:
    """
    Encode aligned frames to an MP4 file.

    progress_callback(current, total) is called on the encoding thread.
    status_callback(message) is called for post-processing steps (ffmpeg).
    audio_path: optional audio file to mix in (requires ffmpeg).
    Raises on failure.
    """
    if not frames:
        raise ValueError("No frames to export.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Clean up any stale tmp file from a previous failed export.
    tmp_path = output_path.with_suffix(".tmp.mp4")
    tmp_path.unlink(missing_ok=True)

    frame_seq = build_frame_sequence(frames, fps, hold_seconds, transition_seconds)
    total     = len(frame_seq)

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

    # Pass 1 – re-encode mp4v → H.264 for broad player compatibility.
    # Pass 2 – copy the H.264 stream and mix in audio (no re-encode of video).
    video_duration = total / fps
    if status_callback:
        status_callback("Re-encoding to H.264…")
    _try_reencode_h264(output_path)
    if audio_path and Path(audio_path).exists():
        if status_callback:
            status_callback("Mixing audio…")
        _mix_audio(output_path, Path(audio_path), video_duration)


def _try_reencode_h264(path: Path) -> None:
    """Re-encode mp4v → H.264 using ffmpeg. No-op if ffmpeg is not on PATH."""
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


def _mix_audio(video_path: Path, audio_path: Path, video_duration: float) -> None:
    """
    Add a looped audio track to an already-encoded video.

    Uses -c:v copy so the video stream is not re-encoded.
    Audio is looped with -stream_loop -1, faded out 2 s before the end,
    and the output is trimmed to the video length with -shortest.
    """
    import subprocess, shutil

    if not shutil.which("ffmpeg"):
        return

    fade_start = max(0.0, video_duration - 2.0)
    tmp = video_path.with_suffix(".tmp.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-stream_loop", "-1", "-i", str(audio_path),
                "-c:v", "copy",
                "-filter_complex",
                f"[1:a]afade=t=out:st={fade_start:.3f}:d=2[a]",
                "-map", "0:v:0",
                "-map", "[a]",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                str(tmp),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=300,
        )
        if result.returncode == 0:
            tmp.replace(video_path)
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
    audio_path: Optional[Path] = None,
    status_callback: Optional[Callable[[str], None]] = None,
) -> threading.Thread:
    """Run export_mp4 on a background thread; returns the Thread."""

    def _run():
        try:
            export_mp4(
                frames, output_path, fps, hold_seconds,
                transition_seconds, progress_callback, audio_path, status_callback,
            )
            if done_callback:
                done_callback(None)
        except Exception as exc:
            if done_callback:
                done_callback(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t
