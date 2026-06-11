"""Shared constants, helpers, and sort utilities used across ui modules."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_AUDIO_EXTS   = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
_MUSIC_DIR    = Path(__file__).parent.parent / "music"
_NO_AUDIO     = "None"
_SORT_OPTIONS = ["Date ↑", "Date ↓", "Filename ↑", "Filename ↓"]
_EXIF_DATE_TAGS = (36867, 306)  # DateTimeOriginal, DateTime

_RESOLUTIONS = {
    "1920×1080": (1920, 1080),
    "1080×1080": (1080, 1080),
    "720×720":   (720, 720),
    "640×640":   (640, 640),
}


def _scan_music() -> dict[str, Path]:
    """Return {filename: path} for audio files in the music/ folder, sorted by name."""
    if not _MUSIC_DIR.exists():
        return {}
    return {
        p.name: p
        for p in sorted(_MUSIC_DIR.iterdir())
        if p.suffix.lower() in _AUDIO_EXTS
    }


def _photo_date(path: Path) -> datetime:
    """Return the best available date for sorting: EXIF → file mtime."""
    try:
        from PIL import Image as _PIL
        with _PIL.open(path) as img:
            exif = img._getexif()
            if exif:
                for tag_id in _EXIF_DATE_TAGS:
                    val = exif.get(tag_id)
                    if val:
                        try:
                            return datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                        except ValueError:
                            pass
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def _sort_images(images: list[Path], sort_option: str) -> list[Path]:
    desc = "↓" in sort_option
    if "Date" in sort_option:
        return sorted(images, key=_photo_date, reverse=desc)
    return sorted(images, key=lambda p: p.name.lower(), reverse=desc)


def _sort_images_keyed(
    pairs: list[tuple[Path, object]], sort_option: str
) -> list[tuple[Path, object]]:
    """Sort (path, anything) pairs by the same key as _sort_images."""
    desc = "↓" in sort_option
    if "Date" in sort_option:
        return sorted(pairs, key=lambda t: _photo_date(t[0]), reverse=desc)
    return sorted(pairs, key=lambda t: t[0].name.lower(), reverse=desc)
