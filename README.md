# Face Sequence Aligner

A desktop app that turns a folder of face photos into a smooth MP4 timelapse. Originally designed so that I could make a video of my child's face changing across years of photos. Each photo is automatically aligned so the eyes always land at the same position in the frame, producing a stable morph effect when the frames are played back.

## Features

- **Automatic face alignment** - detects facial landmarks (eyes) and applies an affine transform so every frame has the eyes at a consistent position and scale
- **Small-face detection** - uses a Haar cascade + MediaPipe crop pipeline to find faces that are a small fraction of a high-resolution image (e.g. Pixel 9 Pro 8K shots)
- **EXIF rotation** - phone photos in portrait orientation are correctly loaded regardless of how the JPEG is stored on disk
- **Multi-face picker** - if a photo contains more than one face, a dialog shows the image with labelled bounding boxes so you can click the right person
- **Context-aware zoom** - the output scale is derived from the median "fit-to-frame" face size across all photos, so the video shows as much surrounding context as possible while keeping faces at a consistent size; adjustable via the **Face %** field
- **Sort order** - photos are ordered by EXIF date (falling back to file date); can be switched to filename ascending/descending or date descending
- **Thumbnail preview** - a scrollable strip of aligned thumbnails lets you review alignment before exporting
- **Music / audio** - optional audio track mixed into the exported video; choose from royalty-free tracks in the `music/` folder or browse for any file; audio loops automatically and fades out at the end
- **MP4 export** - configurable hold duration, cross-fade transition length, and FPS; re-encodes to H.264 via ffmpeg if available

## Requirements

- Python 3.11+
- ffmpeg on `PATH` (optional - required for H.264 output and audio mixing; falls back to mp4v with no audio if absent)

Install Python dependencies:

```
pip install -r requirements.txt
```

## Usage

```
python main.py
```

1. Click **Open Folder…** and select a directory of photos (JPEG, PNG, TIFF, BMP, WebP)
2. Choose a **Sort** order - default is EXIF date ascending (oldest first), falling back to file date when EXIF is absent
3. Set the output **Resolution**, **Hold**, **Transition**, **FPS**, and optionally a **Face %** (see below)
4. Click **Align Photos** - faces are detected and aligned; multi-face photos prompt a picker dialog. Click the face or select with button.
5. Review the thumbnail strip; click a thumbnail to see its filename
6. Click **Export MP4…**, choose a save location, and wait for encoding to complete

## Settings reference

| Setting | Description |
|---|---|
| **Resolution** | Output frame size. Changing this requires re-aligning before export. |
| **Hold (s)** | How long each photo is held in the video before the next cross-fade. |
| **Transition (s)** | Duration of the cross-fade between consecutive photos. |
| **FPS** | Frames per second of the output MP4. |
| **Face %** | Inter-eye distance as a percentage of the frame width. Lower values zoom out, showing more surrounding context; higher values zoom in. Leave blank to auto-compute from the dataset. |
| **Sort** | `Date ↑` (default) - EXIF date ascending, file date fallback. Also available: `Date ↓`, `Filename ↑`, `Filename ↓`. |
| **Audio** | Select a track from the `music/` folder, or click **Browse…** to use any file. Set to `None` for a silent video. Requires ffmpeg. |

## Changing settings without re-aligning

- **Hold, Transition, FPS, Audio** - change freely; just click **Export MP4…** again with the same aligned frames.
- **Face %** - change and click **Align Photos** again; detection is cached so only the warp step re-runs (fast).
- **Sort** - change and click **Align Photos** again; detection cache is reused, only the order and warp step re-run.
- **Resolution** - requires a full re-align; the **Export** button is disabled automatically when the dropdown doesn't match the aligned frames.

## Project structure

```
main.py           - entry point
aligner.py        - face detection (MediaPipe + Haar fallback) and affine warp
exporter.py       - cross-fade frame builder and OpenCV/ffmpeg video encoder
ui/
  app.py          - main window
  face_picker.py  - multi-face selection dialog
  preview.py      - scrollable thumbnail strip widget
music/            - drop royalty-free audio files here; see music/README.md for sources
requirements.txt
diagnose.py       - CLI tool for debugging face detection on a folder of images
```

## Diagnostics

If faces are not being detected, run:

```
python diagnose.py path/to/photos
```

This reports, for each image, the resolution, whether EXIF rotation was applied, and which detection pass found the face (direct MediaPipe or Haar+crop fallback).

## Notes

- The MediaPipe face landmarker model (~30 MB) is downloaded automatically to `~/.cache/face-sequence-aligner/` on first run
- Face detection caches per folder - switching sort order or Face % re-uses cached detections and skips straight to warping
- Audio files in `music/` are gitignored; only the folder's `README.md` is tracked. See `music/README.md` for royalty-free sources
