"""
Wrapping thumbnail grid showing aligned frames.

Each thumbnail is a small square derived from the full-resolution AlignedFrame.
Click to select; drag to reorder.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk, ImageEnhance
import cv2

from aligner import AlignedFrame

THUMB_SIZE = 120
THUMB_PAD  = 6
LABEL_H    = 14   # height reserved below each thumb for the number label
SELECTED_BORDER = 3
DRAG_THRESHOLD  = 8   # pixels before drag mode activates
MARKER_COLOR    = "#4da6ff"

_CELL = THUMB_SIZE + THUMB_PAD   # one grid cell width/height


class PreviewStrip(ctk.CTkFrame):
    """
    Scrollable wrapping grid of aligned-frame thumbnails.

    Callbacks:
        on_select(idx)               — fired when a thumbnail is clicked
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
        self._drag_start_y: float = 0.0
        self._dragging:     bool  = False
        self._drop_insert:  int   = -1   # insertion slot (0 … n)

        self._canvas = ctk.CTkCanvas(
            self,
            highlightthickness=0,
            bg="#1a1a1a",
        )
        self._scrollbar = ctk.CTkScrollbar(
            self, orientation="vertical", command=self._canvas.yview
        )
        self._canvas.configure(yscrollcommand=self._scrollbar.set)
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")

        self._canvas.bind("<Button-1>",       self._on_press)
        self._canvas.bind("<B1-Motion>",      self._on_motion)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<MouseWheel>",     self._on_wheel)
        self._canvas.bind("<Configure>",      self._on_resize)

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
    # Layout helpers
    # ------------------------------------------------------------------

    def _cols(self) -> int:
        """Number of columns that fit in the current canvas width."""
        w = self._canvas.winfo_width()
        if w < 10:
            w = 800  # fallback before first layout pass
        return max(1, (w - THUMB_PAD) // _CELL)

    def _row_h(self) -> int:
        return THUMB_SIZE + THUMB_PAD + LABEL_H

    def _thumb_pos(self, i: int) -> tuple[int, int]:
        """Return (x0, y0) canvas origin for thumbnail i."""
        cols = self._cols()
        x0 = THUMB_PAD + (i % cols) * _CELL
        y0 = THUMB_PAD + (i // cols) * self._row_h()
        return x0, y0

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _draw_empty_hint(self) -> None:
        self._canvas.create_text(
            10, THUMB_SIZE // 2,
            text="Open a folder and click 'Align Photos' — thumbnails will appear here.",
            fill="#555555",
            anchor="w",
            font=("", 11),
        )

    def _render(self) -> None:
        self._canvas.delete("all")
        self._thumb_images.clear()

        n = len(self._frames)
        if n == 0:
            self._draw_empty_hint()
            return

        cols  = self._cols()
        row_h = self._row_h()
        total_h = math.ceil(n / cols) * row_h + THUMB_PAD
        canvas_w = max(self._canvas.winfo_width(), cols * _CELL + THUMB_PAD)
        self._canvas.configure(scrollregion=(0, 0, canvas_w, total_h))

        for i, frame in enumerate(self._frames):
            x0, y0 = self._thumb_pos(i)

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

        # Insertion marker: vertical bar on the left edge of the insertion slot
        if self._dragging and 0 <= self._drop_insert <= n:
            ins = self._drop_insert
            if ins == n:
                # After the last item
                last_col = (n - 1) % cols
                last_row = (n - 1) // cols
                mx  = THUMB_PAD + (last_col + 1) * _CELL - THUMB_PAD // 2
                my0 = THUMB_PAD + last_row * row_h
            else:
                mx  = THUMB_PAD + (ins % cols) * _CELL - THUMB_PAD // 2
                my0 = THUMB_PAD + (ins // cols) * row_h
            self._canvas.create_line(
                mx, my0 - 2, mx, my0 + THUMB_SIZE + 2,
                fill=MARKER_COLOR, width=3,
            )

    def _make_thumb(self, frame: AlignedFrame, dim: bool = False) -> ImageTk.PhotoImage:
        bgr = frame.image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        if dim:
            pil = ImageEnhance.Brightness(pil).enhance(0.4)
        return ImageTk.PhotoImage(pil)

    # ------------------------------------------------------------------
    # Drag-drop helpers
    # ------------------------------------------------------------------

    def _idx_at(self, canvas_x: float, canvas_y: float) -> int:
        """Thumbnail index at canvas coordinates (-1 if none)."""
        cols = self._cols()
        col = int((canvas_x - THUMB_PAD) // _CELL)
        row = int((canvas_y - THUMB_PAD) // self._row_h())
        if col < 0 or col >= cols:
            return -1
        idx = row * cols + col
        return idx if 0 <= idx < len(self._frames) else -1

    def _insert_at(self, canvas_x: float, canvas_y: float) -> int:
        """Insertion slot (0 … n) closest to the given canvas coordinates."""
        cols  = self._cols()
        row_h = self._row_h()
        n     = len(self._frames)
        row = max(0, int((canvas_y - THUMB_PAD // 2) // row_h))
        col = max(0, min(cols, round((canvas_x - THUMB_PAD / 2) / _CELL)))
        return max(0, min(n, row * cols + col))

    def _reset_drag(self) -> None:
        self._drag_src    = -1
        self._dragging    = False
        self._drop_insert = -1

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_press(self, event) -> None:
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        idx = self._idx_at(cx, cy)
        if idx >= 0:
            self._drag_src     = idx
            self._drag_start_x = cx
            self._drag_start_y = cy
            self._dragging     = False
            self._drop_insert  = -1

    def _on_motion(self, event) -> None:
        if self._drag_src < 0:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        if not self._dragging:
            dx = cx - self._drag_start_x
            dy = cy - self._drag_start_y
            if (dx * dx + dy * dy) ** 0.5 < DRAG_THRESHOLD:
                return
            self._dragging = True

        self._drop_insert = self._insert_at(cx, cy)
        self._render()

    def _on_release(self, event) -> None:
        if self._drag_src < 0:
            return

        if not self._dragging:
            idx = self._drag_src
            self._selected = idx
            self._reset_drag()
            self._render()
            if self._on_select:
                self._on_select(idx)
            return

        src = self._drag_src
        ins = self._drop_insert

        if ins not in (src, src + 1) and 0 <= src < len(self._frames):
            new_idx = ins if ins <= src else ins - 1
            if self._on_reorder:
                self._on_reorder(src, new_idx)
            self._selected = new_idx

        self._reset_drag()
        self._render()

    def _on_wheel(self, event) -> None:
        self._canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")

    def _on_resize(self, event) -> None:
        if self._frames:
            self._render()
