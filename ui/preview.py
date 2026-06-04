"""
Horizontal scrollable thumbnail strip showing aligned frames.

Each thumbnail is a small square derived from the full-resolution AlignedFrame.
Click to select; drag horizontally to reorder.
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2

from aligner import AlignedFrame

THUMB_SIZE = 120
THUMB_PAD  = 6
SELECTED_BORDER = 3
DRAG_THRESHOLD  = 8   # pixels of motion before drag mode activates
MARKER_COLOR    = "#4da6ff"


class PreviewStrip(ctk.CTkFrame):
    """
    Scrollable horizontal strip of aligned-frame thumbnails.

    Callbacks:
        on_select(idx)              — fired when a thumbnail is clicked
        on_reorder(old_idx, new_idx) — fired when a drag-drop reorder completes
    """

    def __init__(
        self,
        parent,
        on_select:  Optional[Callable[[int], None]] = None,
        on_reorder: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self._on_select  = on_select
        self._on_reorder = on_reorder
        self._frames: list[AlignedFrame] = []
        self._thumb_images: list[ImageTk.PhotoImage] = []
        self._selected: int = -1

        # Drag state
        self._drag_src:     int   = -1
        self._drag_start_x: float = 0.0
        self._dragging:     bool  = False
        self._drop_insert:  int   = -1   # insertion slot (0 … n)

        self._canvas = ctk.CTkCanvas(
            self,
            height=THUMB_SIZE + THUMB_PAD * 2 + 20,
            highlightthickness=0,
            bg="#1a1a1a",
        )
        self._scrollbar = ctk.CTkScrollbar(
            self, orientation="horizontal", command=self._canvas.xview
        )
        self._canvas.configure(xscrollcommand=self._scrollbar.set)
        self._canvas.pack(fill="x", expand=True)
        self._scrollbar.pack(fill="x")

        self._canvas.bind("<Button-1>",        self._on_press)
        self._canvas.bind("<B1-Motion>",        self._on_motion)
        self._canvas.bind("<ButtonRelease-1>",  self._on_release)
        self._canvas.bind("<MouseWheel>",       self._on_wheel)

        self._draw_empty_hint()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_frames(self, frames: list[AlignedFrame]) -> None:
        self._frames = list(frames)
        self._selected = -1
        self._reset_drag()
        self._render()

    def clear(self) -> None:
        self._frames = []
        self._thumb_images.clear()
        self._reset_drag()
        self._canvas.delete("all")
        self._canvas.configure(scrollregion=(0, 0, 0, 0))
        self._draw_empty_hint()

    def select(self, index: int) -> None:
        if 0 <= index < len(self._frames):
            self._selected = index
            self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_empty_hint(self) -> None:
        self._canvas.create_text(
            10, (THUMB_SIZE + THUMB_PAD * 2) // 2,
            text="Open a folder and click 'Align Photos' — thumbnails will appear here.",
            fill="#555555",
            anchor="w",
            font=("", 11),
        )

    def _render(self) -> None:
        self._canvas.delete("all")
        self._thumb_images.clear()

        n = len(self._frames)
        total_w = n * (THUMB_SIZE + THUMB_PAD) + THUMB_PAD
        strip_h = THUMB_SIZE + THUMB_PAD * 2
        self._canvas.configure(scrollregion=(0, 0, total_w, strip_h))

        for i, frame in enumerate(self._frames):
            x0 = THUMB_PAD + i * (THUMB_SIZE + THUMB_PAD)
            y0 = THUMB_PAD

            # Dim the thumbnail being dragged
            thumb = self._make_thumb(frame, dim=(self._dragging and i == self._drag_src))
            self._thumb_images.append(thumb)
            self._canvas.create_image(x0, y0, anchor="nw", image=thumb)

            if i == self._selected and not self._dragging:
                self._canvas.create_rectangle(
                    x0 - SELECTED_BORDER, y0 - SELECTED_BORDER,
                    x0 + THUMB_SIZE + SELECTED_BORDER, y0 + THUMB_SIZE + SELECTED_BORDER,
                    outline=MARKER_COLOR, width=SELECTED_BORDER,
                )

            self._canvas.create_text(
                x0 + THUMB_SIZE // 2, y0 + THUMB_SIZE + 2,
                text=str(i + 1), fill="#aaaaaa", font=("", 9),
            )

        # Insertion marker
        if self._dragging and 0 <= self._drop_insert <= n:
            ins = self._drop_insert
            marker_x = THUMB_PAD + ins * (THUMB_SIZE + THUMB_PAD) - THUMB_PAD // 2
            self._canvas.create_line(
                marker_x, THUMB_PAD - 2,
                marker_x, THUMB_PAD + THUMB_SIZE + 2,
                fill=MARKER_COLOR, width=3,
            )

    def _make_thumb(self, frame: AlignedFrame, dim: bool = False) -> ImageTk.PhotoImage:
        bgr = frame.image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        if dim:
            from PIL import ImageEnhance
            pil = ImageEnhance.Brightness(pil).enhance(0.4)
        return ImageTk.PhotoImage(pil)

    # ------------------------------------------------------------------
    # Drag-drop helpers
    # ------------------------------------------------------------------

    def _idx_at(self, canvas_x: float) -> int:
        """Thumbnail index under canvas_x (-1 if none)."""
        idx = int((canvas_x - THUMB_PAD) // (THUMB_SIZE + THUMB_PAD))
        return idx if 0 <= idx < len(self._frames) else -1

    def _insert_at(self, canvas_x: float) -> int:
        """Insertion slot (0 … n) closest to canvas_x."""
        raw = (canvas_x - THUMB_PAD / 2) / (THUMB_SIZE + THUMB_PAD)
        return max(0, min(len(self._frames), round(raw)))

    def _reset_drag(self) -> None:
        self._drag_src    = -1
        self._dragging    = False
        self._drop_insert = -1

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        cx = self._canvas.canvasx(event.x)
        idx = self._idx_at(cx)
        if idx >= 0:
            self._drag_src    = idx
            self._drag_start_x = cx
            self._dragging    = False
            self._drop_insert = -1

    def _on_motion(self, event) -> None:
        if self._drag_src < 0:
            return
        cx = self._canvas.canvasx(event.x)
        if not self._dragging:
            if abs(cx - self._drag_start_x) < DRAG_THRESHOLD:
                return
            self._dragging = True

        self._drop_insert = self._insert_at(cx)
        self._render()

    def _on_release(self, event) -> None:
        if self._drag_src < 0:
            return

        if not self._dragging:
            # Treat as a regular click
            idx = self._drag_src
            self._selected = idx
            self._reset_drag()
            self._render()
            if self._on_select:
                self._on_select(idx)
            return

        src = self._drag_src
        ins = self._drop_insert

        # ins == src or ins == src+1 means dropped on itself — no-op
        if ins not in (src, src + 1) and 0 <= src < len(self._frames):
            new_idx = ins if ins <= src else ins - 1
            if self._on_reorder:
                self._on_reorder(src, new_idx)
            # Update local selection to follow the moved item
            self._selected = new_idx

        self._reset_drag()
        self._render()

    def _on_wheel(self, event) -> None:
        self._canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
