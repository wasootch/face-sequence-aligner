"""
Main application window.

Layout (top → bottom):
  Toolbar   — folder picker (row 0), settings + action buttons (row 1)
  Preview   — aligned-frame thumbnail strip
  Status    — progress bar + status label
"""

from __future__ import annotations

import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np
import customtkinter as ctk

from aligner import AlignedFrame, FaceAligner
from exporter import export_mp4_async
from ui.preview import PreviewStrip
from ui.face_picker import pick_face

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

_SORT_OPTIONS = ["Date ↑", "Date ↓", "Filename ↑", "Filename ↓"]

_EXIF_DATE_TAGS = (36867, 306)   # DateTimeOriginal, DateTime


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
    """Sort (path, face) pairs using the same key as _sort_images."""
    desc = "↓" in sort_option
    if "Date" in sort_option:
        return sorted(pairs, key=lambda t: _photo_date(t[0]), reverse=desc)
    return sorted(pairs, key=lambda t: t[0].name.lower(), reverse=desc)


_RESOLUTIONS = {
    "1920×1080": (1920, 1080),
    "1080×1080": (1080, 1080),
    "720×720": (720, 720),
    "640×640": (640, 640),
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Face Sequence Aligner")
        self.geometry("1100x680")
        self.minsize(800, 500)

        self._folder: Optional[Path] = None
        self._aligned_frames: list[AlignedFrame] = []
        self._aligner: Optional[FaceAligner] = None
        self._export_thread: Optional[threading.Thread] = None

        # Cached face detections — reused when only Face % changes so the
        # slow detection step can be skipped on subsequent Align runs.
        self._pending_faces: list[tuple[Path, object]] = []
        self._detected_folder: Optional[Path] = None

        # Resolution the current _aligned_frames were rendered at.
        self._aligned_output_size: Optional[tuple[int, int]] = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(self, corner_radius=0)
        toolbar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self._build_toolbar(toolbar)

        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 0))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self._preview = PreviewStrip(
            preview_frame,
            on_select=self._on_thumbnail_select,
            on_reorder=self._on_thumbnail_reorder,
        )
        self._preview.grid(row=0, column=0, sticky="nsew")

        self._detail_label = ctk.CTkLabel(preview_frame, text="")
        self._detail_label.grid(row=1, column=0, pady=4)

        status_bar = ctk.CTkFrame(self, corner_radius=0, height=48)
        status_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        status_bar.grid_propagate(False)
        self._build_status_bar(status_bar)

    def _build_toolbar(self, parent: ctk.CTkFrame):
        parent.grid_columnconfigure(1, weight=1)  # folder label stretches

        # --- Row 0: folder + sort ---
        ctk.CTkButton(
            parent, text="Open Folder…", command=self._pick_folder, width=130
        ).grid(row=0, column=0, padx=(10, 6), pady=(8, 2), sticky="w")

        self._folder_label = ctk.CTkLabel(parent, text="No folder selected", anchor="w")
        self._folder_label.grid(row=0, column=1, padx=(0, 10), pady=(8, 2), sticky="ew")

        ctk.CTkLabel(parent, text="Sort:").grid(row=0, column=2, padx=(0, 4), pady=(8, 2))
        self._sort_var = ctk.StringVar(value="Date ↑")
        ctk.CTkOptionMenu(
            parent, variable=self._sort_var, values=_SORT_OPTIONS, width=130,
        ).grid(row=0, column=3, padx=(0, 10), pady=(8, 2))

        # --- Row 1: settings ---
        r1 = ctk.CTkFrame(parent, fg_color="transparent")
        r1.grid(row=1, column=0, columnspan=99, sticky="ew", padx=10, pady=(2, 8))

        col = 0

        ctk.CTkLabel(r1, text="Resolution:").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._res_var = ctk.StringVar(value="1080×1080")
        self._res_var.trace_add("write", lambda *_: self._on_resolution_changed())
        ctk.CTkOptionMenu(
            r1, variable=self._res_var, values=list(_RESOLUTIONS.keys()), width=120
        ).grid(row=0, column=col, padx=(0, 16))
        col += 1

        ctk.CTkLabel(r1, text="Hold (s):").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._hold_var = ctk.StringVar(value="1.5")
        ctk.CTkEntry(r1, textvariable=self._hold_var, width=52).grid(row=0, column=col, padx=(0, 16))
        col += 1

        ctk.CTkLabel(r1, text="Transition (s):").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._trans_var = ctk.StringVar(value="1.0")
        ctk.CTkEntry(r1, textvariable=self._trans_var, width=52).grid(row=0, column=col, padx=(0, 16))
        col += 1

        ctk.CTkLabel(r1, text="FPS:").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._fps_var = ctk.StringVar(value="30")
        ctk.CTkEntry(r1, textvariable=self._fps_var, width=44).grid(row=0, column=col, padx=(0, 16))
        col += 1

        ctk.CTkLabel(r1, text="Face %:").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._face_pct_var = ctk.StringVar(value="")
        ctk.CTkEntry(
            r1, textvariable=self._face_pct_var, width=52, placeholder_text="auto"
        ).grid(row=0, column=col, padx=(0, 16))
        col += 1

        r1.grid_columnconfigure(col, weight=1)
        col += 1

        self._align_btn = ctk.CTkButton(
            r1, text="Align Photos", command=self._start_align, width=120
        )
        self._align_btn.grid(row=0, column=col, padx=(0, 8))
        col += 1

        self._export_btn = ctk.CTkButton(
            r1, text="Export MP4…", command=self._start_export, width=120, state="disabled"
        )
        self._export_btn.grid(row=0, column=col)

    def _build_status_bar(self, parent: ctk.CTkFrame):
        parent.grid_columnconfigure(0, weight=1)

        self._status_label = ctk.CTkLabel(parent, text="Ready.", anchor="w")
        self._status_label.grid(row=0, column=0, padx=12, pady=4, sticky="w")

        self._progress = ctk.CTkProgressBar(parent, width=300)
        self._progress.set(0)
        self._progress.grid(row=0, column=1, padx=(0, 12), pady=4)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _pick_folder(self):
        folder = filedialog.askdirectory(title="Select photo folder")
        if not folder:
            return
        self._folder = Path(folder)
        self._folder_label.configure(text=str(self._folder))
        self._aligned_frames.clear()
        self._preview.clear()
        self._export_btn.configure(state="disabled")
        # Invalidate cached detections — new folder means new images.
        self._pending_faces.clear()
        self._detected_folder = None
        self._face_pct_var.set("")
        self._set_status("Folder selected. Click 'Align Photos' to process.")

    def _start_align(self):
        if not self._folder:
            messagebox.showwarning("No folder", "Please open a photo folder first.")
            return

        images = sorted(
            p for p in self._folder.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            messagebox.showerror("No images", "No supported image files found in the folder.")
            return

        res_key = self._res_var.get()
        output_size = _RESOLUTIONS.get(res_key, (1080, 1080))

        # If we have cached detections for this folder, skip Phase 1 (detection)
        # and only re-run the warp.  This makes changing Face % very fast.
        can_skip_detection = (
            bool(self._pending_faces) and self._detected_folder == self._folder
        )

        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._aligned_frames.clear()
        self._preview.clear()
        self._progress.set(0)

        if can_skip_detection:
            self._set_status("Re-aligning with new face size (skipping detection)…")
        else:
            self._set_status(f"Detecting faces in {len(images)} photo(s)…")

        threading.Thread(
            target=self._align_worker,
            args=(images, output_size, can_skip_detection, self._sort_var.get()),
            daemon=True,
        ).start()

    def _align_worker(
        self,
        images: list[Path],
        output_size: tuple[int, int],
        skip_detection: bool,
        sort_option: str,
    ):
        if self._aligner:
            self._aligner.close()
        self._aligner = FaceAligner(output_size=output_size)

        skipped = 0

        # --- Phase 1: detect faces (skipped when only Face % changed) ---
        if skip_detection:
            pending = self._pending_faces
        else:
            pending = []
            n = len(images)
            for i, path in enumerate(images):
                self._set_status(f"Detecting [{i+1}/{n}] {path.name}")
                self._progress.set((i + 1) / (n * 2))

                try:
                    faces = self._aligner.detect_faces(path)
                except Exception as exc:
                    self._set_status(f"Error reading {path.name}: {exc}")
                    skipped += 1
                    continue

                if not faces:
                    self._set_status(f"No face found in {path.name} — skipped.")
                    skipped += 1
                    continue

                if len(faces) == 1:
                    chosen = faces[0]
                else:
                    chosen = self._ask_face_pick(path, faces)
                    if chosen is None:
                        skipped += 1
                        continue

                pending.append((path, chosen))

            self._pending_faces = pending
            self._detected_folder = self._folder

        if not pending:
            self.after(0, self._align_done, [], skipped, None)
            return

        # Sort a copy so the cached _pending_faces order is never mutated.
        pending = _sort_images_keyed(pending, sort_option)

        # --- Phase 2: determine target eye size ---
        output_w, _ = output_size

        # If the user typed a Face % value, use it directly.
        target_eye_px: Optional[float] = None
        try:
            user_pct = float(self._face_pct_var.get())
            if 1.0 <= user_pct <= 80.0:
                target_eye_px = user_pct / 100.0 * output_w
        except ValueError:
            pass  # empty / "auto" — compute from data below

        if target_eye_px is None:
            # Auto: median of how large each face would be at fit-to-frame scale,
            # clamped so faces stay between 5% and 25% of output width.
            from PIL import Image as _PIL, ImageOps as _IOps
            fit_eye_pxs: list[float] = []
            for path, face in pending:
                try:
                    with _PIL.open(path) as _img:
                        _img = _IOps.exif_transpose(_img)
                        orig_w, orig_h = _img.size
                    fit_eye_pxs.append(self._aligner.eye_px_at_fit(face, orig_w, orig_h))
                except Exception:
                    pass

            if fit_eye_pxs:
                raw = float(np.median(fit_eye_pxs))
                target_eye_px = float(np.clip(raw, 0.05 * output_w, 0.25 * output_w))
            else:
                target_eye_px = 0.20 * output_w

        # --- Phase 3: warp every image with the shared target ---
        results: list[AlignedFrame] = []
        m = len(pending)
        for j, (path, face) in enumerate(pending):
            self._set_status(f"Aligning [{j+1}/{m}] {path.name}")
            self._progress.set(0.5 + (j + 1) / (m * 2))
            try:
                frame = self._aligner.align(path, face, target_eye_px=target_eye_px)
                results.append(frame)
            except Exception as exc:
                self._set_status(f"Align error {path.name}: {exc}")
                skipped += 1

        self.after(0, self._align_done, results, skipped,
                   target_eye_px / output_w * 100, output_size)

    def _ask_face_pick(self, path: Path, faces) -> Optional[object]:
        result_holder = [None]
        event = threading.Event()

        def _show():
            result_holder[0] = pick_face(self, path, faces)
            event.set()

        self.after(0, _show)
        event.wait()
        return result_holder[0]

    def _align_done(
        self,
        results: list[AlignedFrame],
        skipped: int,
        used_face_pct: Optional[float],
        output_size: Optional[tuple[int, int]] = None,
    ):
        self._aligned_frames = results
        self._aligned_output_size = output_size
        self._preview.set_frames(results)
        self._align_btn.configure(state="normal")

        if used_face_pct is not None:
            # Show the value that was actually used so the user can adjust it.
            self._face_pct_var.set(f"{used_face_pct:.1f}")

        msg = f"Aligned {len(results)} photo(s)."
        if skipped:
            msg += f" {skipped} skipped."
        if used_face_pct is not None:
            msg += f"  Face size: {used_face_pct:.1f}%"
        self._set_status(msg)
        self._progress.set(1.0)

        if results:
            self._export_btn.configure(state="normal")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _start_export(self):
        if not self._aligned_frames:
            return

        out_path = filedialog.asksaveasfilename(
            title="Save MP4",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
        )
        if not out_path:
            return

        try:
            fps = int(self._fps_var.get())
            hold = float(self._hold_var.get())
            trans = float(self._trans_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "FPS, hold, and transition must be numbers.")
            return

        self._export_btn.configure(state="disabled")
        self._align_btn.configure(state="disabled")
        self._set_status("Exporting…")
        self._progress.set(0)

        self._export_thread = export_mp4_async(
            frames=self._aligned_frames,
            output_path=Path(out_path),
            fps=fps,
            hold_seconds=hold,
            transition_seconds=trans,
            progress_callback=self._export_progress,
            done_callback=self._export_done,
        )

    def _export_progress(self, current: int, total: int):
        ratio = current / total
        msg = f"Exporting frame {current}/{total}…"
        self.after(0, lambda: (self._progress.set(ratio), self._set_status(msg)))

    def _export_done(self, error: Optional[Exception]):
        def _on_main_thread():
            if error:
                messagebox.showerror("Export failed", str(error))
                self._set_status("Export failed.")
            else:
                self._set_status("Export complete!")
                self._progress.set(1.0)
            self._export_btn.configure(state="normal")
            self._align_btn.configure(state="normal")

        self.after(0, _on_main_thread)

    # ------------------------------------------------------------------

    def _on_thumbnail_select(self, idx: int):
        frame = self._aligned_frames[idx]
        self._detail_label.configure(text=frame.source_path.name)

    def _on_thumbnail_reorder(self, old_idx: int, new_idx: int):
        # Move the aligned frame
        frame = self._aligned_frames.pop(old_idx)
        self._aligned_frames.insert(new_idx, frame)

        # Keep _pending_faces in the same order so that re-aligning
        # (e.g. to change Face %) preserves the user's manual ordering.
        if len(self._pending_faces) == len(self._aligned_frames):
            path_to_pair = {path: face for path, face in self._pending_faces}
            self._pending_faces = [
                (f.source_path, path_to_pair[f.source_path])
                for f in self._aligned_frames
                if f.source_path in path_to_pair
            ]

        self._preview.set_frames(self._aligned_frames)

    def _on_resolution_changed(self):
        selected = _RESOLUTIONS.get(self._res_var.get(), (1080, 1080))
        if self._aligned_output_size and selected != self._aligned_output_size:
            self._export_btn.configure(state="disabled")
            self._set_status(
                f"Resolution changed to {selected[0]}×{selected[1]} — "
                "click 'Align Photos' to update frames before exporting."
            )

    def _set_status(self, msg: str):
        self._status_label.configure(text=msg)

    def on_close(self):
        if self._aligner:
            self._aligner.close()
        self.destroy()
