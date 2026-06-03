"""
Run from the project root:
    python diagnose.py path/to/photo/folder
"""

import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageOps

from aligner import _load_bgr, _ensure_model, _MEDIAPIPE_CAP, _HAAR_PAD

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def diagnose(folder: Path):
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _IMAGE_EXTS)
    if not images:
        print("No images found.")
        return

    model_path = _ensure_model()
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=10,
        min_face_detection_confidence=0.1,
        min_face_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    print(f"\n{'FILE':<45} {'SIZE':>12}  {'RESULT'}")
    print("-" * 90)

    for path in images:
        pil_raw = Image.open(path)
        raw_w, raw_h = pil_raw.size
        pil_corrected = ImageOps.exif_transpose(pil_raw.convert("RGB"))
        cor_w, cor_h = pil_corrected.size
        rotated = " [EXIF rotated]" if (raw_w, raw_h) != (cor_w, cor_h) else ""
        size_str = f"{cor_w}×{cor_h}{rotated}"

        bgr = cv2.cvtColor(np.array(pil_corrected), cv2.COLOR_RGB2BGR)
        w, h = cor_w, cor_h

        result_str = "NO FACE FOUND"

        # Pass 1: MediaPipe on capped image
        scale = min(_MEDIAPIPE_CAP / max(w, h), 1.0)
        det_img = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else bgr
        rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)
        mp_result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if mp_result.face_landmarks:
            n = len(mp_result.face_landmarks)
            dw, dh = det_img.shape[1], det_img.shape[0]
            result_str = f"FOUND {n} face(s) — MediaPipe at {dw}×{dh}"
        else:
            # Pass 2: Haar cascade → crop
            haar_scale = min(_MEDIAPIPE_CAP / max(w, h), 1.0)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            small_gray = cv2.resize(gray, (int(w * haar_scale), int(h * haar_scale)),
                                    interpolation=cv2.INTER_AREA) if haar_scale < 1.0 else gray
            raw_bboxes = haar.detectMultiScale(small_gray, scaleFactor=1.05, minNeighbors=3,
                                               minSize=(24, 24), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(raw_bboxes):
                found_crops = 0
                for xi, yi, wi, hi in raw_bboxes:
                    x  = int(xi / haar_scale)
                    y  = int(yi / haar_scale)
                    fw = int(wi / haar_scale)
                    fh = int(hi / haar_scale)
                    pad = int(max(fw, fh) * _HAAR_PAD)
                    x1, y1 = max(0, x - pad), max(0, y - pad)
                    x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)
                    crop = bgr[y1:y2, x1:x2]
                    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop))
                    if crop_result.face_landmarks:
                        found_crops += 1
                if found_crops:
                    result_str = f"FOUND {found_crops} face(s) — Haar+crop ({len(raw_bboxes)} Haar bbox(es))"
                else:
                    result_str = f"NO FACE — Haar found {len(raw_bboxes)} region(s) but MediaPipe failed on crops"
            else:
                result_str = "NO FACE — Haar found nothing either"

        print(f"{path.name:<45} {size_str:>20}  {result_str}")

    landmarker.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose.py <folder>")
        sys.exit(1)
    diagnose(Path(sys.argv[1]))
