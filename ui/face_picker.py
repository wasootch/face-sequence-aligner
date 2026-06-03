"""
Multi-face selection dialog.

When a photo contains more than one detected face, this dialog shows the full
image with colored bounding boxes drawn around each face. The user clicks a box
(or clicks a numbered button) to select which face to use.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import customtkinter as ctk
from PIL import Image, ImageTk

from aligner import Face

# Box colours per face index (BGR for OpenCV, then converted for display)
_BOX_COLORS_BGR = [
    (0, 200, 255),   # orange
    (0, 255, 100),   # green
    (255, 80, 80),   # blue
    (80, 80, 255),   # red
    (255, 0, 200),   # purple
]


class FacePickerDialog(ctk.CTkToplevel):
    """
    Modal dialog that asks the user to pick one face from a multi-face image.

    After the dialog closes, read `.chosen_face` (a Face or None if cancelled).
    """

    MAX_DISPLAY = 800  # max pixels for the preview image (longest side)

    def __init__(self, parent, image_path: Path, faces: list[Face]):
        super().__init__(parent)
        self.title(f"Select face — {image_path.name}")
        self.resizable(False, False)
        self.grab_set()  # modal

        self.chosen_face: Optional[Face] = None
        self._faces = faces
        self._image_path = image_path

        # --- Build annotated preview image ---
        bgr = cv2.imread(str(image_path))
        annotated = self._annotate(bgr, faces)
        self._tk_image, self._scale = self._fit(annotated)

        # --- Layout ---
        self._canvas = ctk.CTkCanvas(
            self,
            width=self._tk_image.width(),
            height=self._tk_image.height(),
            highlightthickness=0,
        )
        self._canvas.pack(padx=10, pady=10)
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_image)
        self._canvas.bind("<Button-1>", self._on_click)

        label = ctk.CTkLabel(self, text="Click a face or press a numbered button:")
        label.pack()

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=(0, 10))
        for face in faces:
            color = self._box_hex(face.index)
            btn = ctk.CTkButton(
                btn_frame,
                text=f"Face {face.index + 1}",
                fg_color=color,
                command=lambda f=face: self._select(f),
                width=100,
            )
            btn.pack(side="left", padx=4)

        cancel_btn = ctk.CTkButton(self, text="Skip photo", command=self._cancel, width=120)
        cancel_btn.pack(pady=(0, 10))

        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_window()

    # ------------------------------------------------------------------

    def _annotate(self, bgr, faces: list[Face]) -> "np.ndarray":
        import numpy as np
        img = bgr.copy()
        for face in faces:
            color = _BOX_COLORS_BGR[face.index % len(_BOX_COLORS_BGR)]
            x, y, w, h = face.bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
            cv2.putText(
                img, str(face.index + 1),
                (x + 6, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3,
            )
        return img

    def _fit(self, bgr) -> tuple["ImageTk.PhotoImage", float]:
        """Resize BGR image to fit MAX_DISPLAY on its longest side."""
        h, w = bgr.shape[:2]
        scale = min(self.MAX_DISPLAY / w, self.MAX_DISPLAY / h, 1.0)
        nw, nh = int(w * scale), int(h * scale)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
        return ImageTk.PhotoImage(pil), scale

    def _on_click(self, event):
        # Map display coords back to original image, find which bbox was hit
        ox = event.x / self._scale
        oy = event.y / self._scale
        for face in self._faces:
            x, y, w, h = face.bbox
            if x <= ox <= x + w and y <= oy <= y + h:
                self._select(face)
                return

    def _select(self, face: Face):
        self.chosen_face = face
        self.destroy()

    def _cancel(self):
        self.chosen_face = None
        self.destroy()

    @staticmethod
    def _box_hex(idx: int) -> str:
        colors = ["#FF8800", "#00CC44", "#4466FF", "#FF2222", "#CC00CC"]
        return colors[idx % len(colors)]


def pick_face(parent, image_path: Path, faces: list[Face]) -> Optional[Face]:
    """
    Convenience wrapper: opens FacePickerDialog and returns the chosen Face,
    or None if the user cancelled / skipped.
    """
    dlg = FacePickerDialog(parent, image_path, faces)
    return dlg.chosen_face
