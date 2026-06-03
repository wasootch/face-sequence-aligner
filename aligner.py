"""
Face detection and affine alignment.

Detection strategy (in order, stops at first success):
  1. MediaPipe on image scaled to ≤1920px longest side
  2. Haar cascade → crop around each rough bbox → MediaPipe on crop
     (catches small faces that are lost in MediaPipe's internal downscale)

Landmarks are always returned in original-image pixel coordinates.
"""

import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from PIL import Image, ImageOps

_LEFT_EYE_INDICES  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

_EYE_LEFT_X  = 0.35
_EYE_RIGHT_X = 0.65
_EYE_Y       = 0.40

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
_MODEL_CACHE = Path.home() / ".cache" / "face-sequence-aligner" / "face_landmarker.task"

_MEDIAPIPE_CAP = 1920   # longest side cap for MediaPipe detection pass
_HAAR_PAD      = 0.8    # padding around each Haar bbox (fraction of bbox size)
_NMS_IOU       = 0.4    # IoU threshold for deduplicating Haar+crop results


def _load_bgr(image_path: Path) -> np.ndarray:
    """Load image as BGR with EXIF rotation applied."""
    pil = ImageOps.exif_transpose(Image.open(image_path).convert("RGB"))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _ensure_model() -> Path:
    if _MODEL_CACHE.exists():
        return _MODEL_CACHE
    _MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe face landmarker model to {_MODEL_CACHE} …")
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_CACHE)
    return _MODEL_CACHE


@dataclass
class Face:
    index: int
    bbox: tuple[int, int, int, int]   # (x, y, w, h) in original image pixels
    left_eye: tuple[float, float]     # eye centre in original image pixels
    right_eye: tuple[float, float]


@dataclass
class AlignedFrame:
    source_path: Path
    image: np.ndarray       # aligned BGR at output resolution
    face: Face
    transform: np.ndarray   # 2×3 affine matrix


def _iou(a: Face, b: Face) -> float:
    ax, ay, aw, ah = a.bbox
    bx, by, bw, bh = b.bbox
    ix = max(ax, bx);  iy = max(ay, by)
    iw = max(0, min(ax + aw, bx + bw) - ix)
    ih = max(0, min(ay + ah, by + bh) - iy)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _nms(faces: list[Face], iou_threshold: float = _NMS_IOU) -> list[Face]:
    """Remove overlapping duplicate detections; largest bbox wins ties."""
    faces = sorted(faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
    kept: list[Face] = []
    for face in faces:
        if not any(_iou(face, k) > iou_threshold for k in kept):
            kept.append(face)
    for i, f in enumerate(kept):
        f.index = i
    kept.sort(key=lambda f: f.bbox[0])
    return kept


class FaceAligner:
    def __init__(self, output_size: tuple[int, int] = (1080, 1080)):
        self.output_w, self.output_h = output_size

        model_path = _ensure_model()
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=10,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        self._haar = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image_path: Path) -> list[Face]:
        return self._detect_in_bgr(_load_bgr(image_path))

    def align(self, image_path: Path, face: Face) -> AlignedFrame:
        bgr = _load_bgr(image_path)
        aligned, matrix = self._affine_warp(bgr, face)
        return AlignedFrame(source_path=image_path, image=aligned, face=face, transform=matrix)

    def detect_and_align(self, image_path: Path) -> list[AlignedFrame]:
        bgr = _load_bgr(image_path)
        faces = self._detect_in_bgr(bgr)
        results = []
        for f in faces:
            warped, matrix = self._affine_warp(bgr, f)
            results.append(AlignedFrame(source_path=image_path, image=warped, face=f, transform=matrix))
        return results

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _detect_in_bgr(self, bgr: np.ndarray) -> list[Face]:
        h, w = bgr.shape[:2]

        # Pass 1: MediaPipe on a size-capped image.
        # Coordinates are converted back using the ORIGINAL w/h so that landmarks
        # are always in original-image pixel space regardless of detection scale.
        scale = min(_MEDIAPIPE_CAP / max(w, h), 1.0)
        det_img = (
            cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else bgr
        )
        faces = self._run_mediapipe(det_img, ref_w=w, ref_h=h)
        if faces:
            return faces

        # Pass 2: Haar cascade finds rough bboxes that survive MediaPipe's
        # internal downscale; we crop each region and run MediaPipe on the crop.
        return self._haar_crop_detect(bgr, w, h)

    def _run_mediapipe(
        self,
        img: np.ndarray,
        ref_w: int | None = None,
        ref_h: int | None = None,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[Face]:
        """
        Run the FaceLandmarker on `img`.

        ref_w / ref_h: the coordinate space to convert normalised landmarks into.
          - Pass 1 (downscaled whole image): supply original w/h so coords are
            in original-image space.
          - Pass 2 (crop, no downscale): omit; img dimensions + offset are correct.
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.face_landmarks:
            return []

        iw = ref_w if ref_w is not None else img.shape[1]
        ih = ref_h if ref_h is not None else img.shape[0]

        faces: list[Face] = []
        for idx, lm_list in enumerate(result.face_landmarks):
            left_eye  = self._eye_centre(lm_list, _LEFT_EYE_INDICES,  iw, ih, offset_x, offset_y)
            right_eye = self._eye_centre(lm_list, _RIGHT_EYE_INDICES, iw, ih, offset_x, offset_y)
            bbox      = self._landmark_bbox(lm_list, iw, ih, offset_x, offset_y)
            faces.append(Face(index=idx, bbox=bbox, left_eye=left_eye, right_eye=right_eye))

        faces.sort(key=lambda f: f.bbox[0])
        return faces

    def _haar_crop_detect(self, bgr: np.ndarray, w: int, h: int) -> list[Face]:
        """Use Haar to find approximate face regions, then run MediaPipe on crops."""
        haar_scale = min(_MEDIAPIPE_CAP / max(w, h), 1.0)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small_gray = (
            cv2.resize(gray, (int(w * haar_scale), int(h * haar_scale)), interpolation=cv2.INTER_AREA)
            if haar_scale < 1.0 else gray
        )

        raw = self._haar.detectMultiScale(
            small_gray,
            scaleFactor=1.05,
            minNeighbors=4,
            minSize=(24, 24),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if not len(raw):
            return []

        candidates: list[Face] = []
        for xi, yi, wi, hi in raw:
            # Map Haar bbox back to original pixel space
            x  = int(xi / haar_scale)
            y  = int(yi / haar_scale)
            fw = int(wi / haar_scale)
            fh = int(hi / haar_scale)

            pad = int(max(fw, fh) * _HAAR_PAD)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(w, x + fw + pad), min(h, y + fh + pad)

            crop = bgr[y1:y2, x1:x2]
            # No ref_w/ref_h: crop coordinates + offset = original-image coords
            crop_faces = self._run_mediapipe(crop, offset_x=x1, offset_y=y1)
            if not crop_faces:
                continue

            # Keep the one face closest to the Haar bbox centre
            cx, cy = x + fw // 2, y + fh // 2
            best = min(
                crop_faces,
                key=lambda f: abs(f.bbox[0] + f.bbox[2] // 2 - cx)
                            + abs(f.bbox[1] + f.bbox[3] // 2 - cy),
            )
            candidates.append(best)

        return _nms(candidates)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _eye_centre(
        lm_list, indices: list[int],
        iw: int, ih: int,
        off_x: int = 0, off_y: int = 0,
    ) -> tuple[float, float]:
        xs = [lm_list[i].x * iw + off_x for i in indices]
        ys = [lm_list[i].y * ih + off_y for i in indices]
        return (float(np.mean(xs)), float(np.mean(ys)))

    @staticmethod
    def _landmark_bbox(
        lm_list,
        iw: int, ih: int,
        off_x: int = 0, off_y: int = 0,
    ) -> tuple[int, int, int, int]:
        xs = [lm.x * iw + off_x for lm in lm_list]
        ys = [lm.y * ih + off_y for lm in lm_list]
        x0, y0 = int(min(xs)), int(min(ys))
        x1, y1 = int(max(xs)), int(max(ys))
        return (x0, y0, x1 - x0, y1 - y0)

    def _affine_warp(self, bgr: np.ndarray, face: Face) -> tuple[np.ndarray, np.ndarray]:
        lx, ly = face.left_eye
        rx, ry = face.right_eye

        tlx = _EYE_LEFT_X  * self.output_w
        trx = _EYE_RIGHT_X * self.output_w
        ty  = _EYE_Y       * self.output_h

        src_pts = np.float32([
            [lx, ly],
            [rx, ry],
            [lx + (ry - ly), ly - (rx - lx)],
        ])
        dst_pts = np.float32([
            [tlx, ty],
            [trx, ty],
            [tlx, ty - (trx - tlx)],
        ])

        matrix = cv2.getAffineTransform(src_pts, dst_pts)
        warped = cv2.warpAffine(
            bgr, matrix,
            (self.output_w, self.output_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return warped, matrix

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
