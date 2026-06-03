"""
Horizontal scrollable thumbnail strip showing aligned frames.

Each thumbnail is a small square derived from the full-resolution AlignedFrame.
Clicking a thumbnail highlights it and calls an optional on_select callback.
"""

from __future__ import annotations

from typing import Callable, Optional

import customtkinter as ctk
from PIL import Image, ImageTk
import numpy as np
import cv2

from aligner import AlignedFrame

THUMB_SIZE = 120   # pixels (square)
THUMB_PAD = 6      # horizontal gap between thumbs
SELECTED_BORDER = 3


class PreviewStrip(ctk.CTkFrame):
    """
    Scrollable horizontal strip of aligned-frame thumbnails.

    Usage:
        strip = PreviewStrip(parent, on_select=lambda idx: ...)
        strip.set_frames(aligned_frames)
    """

    def __init__(self, parent, on_select: Optional[Callable[[int], None]] = None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_select = on_select
        self._frames: list[AlignedFrame] = []
        self._thumb_images: list[ImageTk.PhotoImage] = []  # keep refs alive
        self._selected: int = -1

        # Scrollable canvas
        self._canvas = ctk.CTkCanvas(
            self,
            height=THUMB_SIZE + THUMB_PAD * 2 + 20,  # +20 for scrollbar
            highlightthickness=0,
            bg="#1a1a1a",
        )
        self._scrollbar = ctk.CTkScrollbar(
            self, orientation="horizontal", command=self._canvas.xview
        )
        self._canvas.configure(xscrollcommand=self._scrollbar.set)

        self._canvas.pack(fill="x", expand=True)
        self._scrollbar.pack(fill="x")

        self._canvas.bind("<Button-1>", self._on_click)
        self._canvas.bind("<MouseWheel>", self._on_wheel)

        self._draw_empty_hint()

    # ------------------------------------------------------------------

    def set_frames(self, frames: list[AlignedFrame]) -> None:
        self._frames = frames
        self._selected = -1
        self._render()

    def clear(self) -> None:
        self._frames = []
        self._thumb_images.clear()
        self._canvas.delete("all")
        self._canvas.configure(scrollregion=(0, 0, 0, 0))
        self._draw_empty_hint()

    def select(self, index: int) -> None:
        if 0 <= index < len(self._frames):
            self._selected = index
            self._render()

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

        total_w = len(self._frames) * (THUMB_SIZE + THUMB_PAD) + THUMB_PAD
        strip_h = THUMB_SIZE + THUMB_PAD * 2
        self._canvas.configure(scrollregion=(0, 0, total_w, strip_h))

        for i, frame in enumerate(self._frames):
            x0 = THUMB_PAD + i * (THUMB_SIZE + THUMB_PAD)
            y0 = THUMB_PAD

            thumb = self._make_thumb(frame)
            self._thumb_images.append(thumb)
            self._canvas.create_image(x0, y0, anchor="nw", image=thumb, tags=f"thumb_{i}")

            if i == self._selected:
                self._canvas.create_rectangle(
                    x0 - SELECTED_BORDER,
                    y0 - SELECTED_BORDER,
                    x0 + THUMB_SIZE + SELECTED_BORDER,
                    y0 + THUMB_SIZE + SELECTED_BORDER,
                    outline="#4da6ff",
                    width=SELECTED_BORDER,
                )

            # Frame number label
            self._canvas.create_text(
                x0 + THUMB_SIZE // 2,
                y0 + THUMB_SIZE + 2,
                text=str(i + 1),
                fill="#aaaaaa",
                font=("", 9),
            )

    def _make_thumb(self, frame: AlignedFrame) -> ImageTk.PhotoImage:
        bgr = frame.image
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb).resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
        return ImageTk.PhotoImage(pil)

    def _on_click(self, event) -> None:
        # Determine which thumbnail was clicked (account for canvas scroll offset)
        cx = self._canvas.canvasx(event.x)
        idx = int((cx - THUMB_PAD) // (THUMB_SIZE + THUMB_PAD))
        if 0 <= idx < len(self._frames):
            self._selected = idx
            self._render()
            if self._on_select:
                self._on_select(idx)

    def _on_wheel(self, event) -> None:
        self._canvas.xview_scroll(-1 if event.delta > 0 else 1, "units")
