# Face Sequence Aligner

A desktop app that turns a folder of face photos into a smooth MP4 timelapse — designed for tracking a child's face changing across years of photos. Each photo is automatically aligned so the eyes always land at the same position in the frame, producing a stable morph effect when the frames are played back.

## Features

- **Automatic face alignment** — detects facial landmarks (eyes) and applies an affine transform so every frame has the eyes at a consistent position and scale
- **Small-face detection** — uses a Haar cascade + MediaPipe crop pipeline to find faces that are a small fraction of a high-resolution image (e.g. Pixel 9 Pro 8K shots)
- **EXIF rotation** — phone photos in portrait orientation are correctly loaded regardless of how the JPEG is stored on disk
- **Multi-face picker** — if a photo contains more than one face, a dialog shows the image with labelled bounding boxes so you can click the right person
- **Context-aware zoom** — the output scale is derived from the median "fit-to-frame" face size across all photos, so the video shows as much surrounding context as possible while keeping faces at a consistent size
- **Thumbnail preview** — a scrollable strip of aligned thumbnails lets you review alignment before exporting
- **MP4 export** — configurable hold duration, cross-fade transition length, and FPS; re-encodes to H.264 via ffmpeg if available

## Requirements

- Python 3.11+
- ffmpeg on `PATH` (optional — improves MP4 compatibility; falls back to mp4v if absent)

Install Python dependencies:

```
pip install -r requirements.txt
```

## Usage

```
python main.py
```

1. Click **Open Folder…** and select a directory of photos (JPEG, PNG, TIFF, BMP, WebP)
2. Set the output resolution, hold duration, transition length, and FPS
3. Click **Align Photos** — faces are detected and aligned; multi-face photos prompt a picker dialog
4. Review the thumbnail strip; click a thumbnail to see its filename
5. Click **Export MP4…**, choose a save location, and wait for encoding to complete
6. You can export multiple times (different durations, FPS) without re-aligning, as long as you don't change the resolution

## Project structure

```
main.py           — entry point
aligner.py        — face detection (MediaPipe + Haar fallback) and affine warp
exporter.py       — cross-fade frame builder and OpenCV/ffmpeg video encoder
ui/
  app.py          — main window
  face_picker.py  — multi-face selection dialog
  preview.py      — scrollable thumbnail strip widget
requirements.txt
diagnose.py       — CLI tool for debugging face detection on a folder of images
```

## Diagnostics

If faces are not being detected, run:

```
python diagnose.py path/to/photos
```

This reports, for each image, the resolution, whether EXIF rotation was applied, and which detection pass found the face (direct MediaPipe or Haar+crop fallback).

## Notes

- Changing the **Resolution** dropdown requires clicking **Align Photos** again — the aligned frames are pre-rendered at the chosen resolution
- Photos are sorted by filename; prefix filenames with dates (e.g. `2018-06-01_birthday.jpg`) for chronological order
- The MediaPipe face landmarker model (~30 MB) is downloaded automatically to `~/.cache/face-sequence-aligner/` on first run
