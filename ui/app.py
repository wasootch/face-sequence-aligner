"""
Main application window.

Layout (top → bottom):
  Toolbar   — folder picker, output resolution, hold/transition controls, Export button
  Preview   — aligned-frame thumbnail strip
  Status    — progress bar + status label
"""

from __future__ import annotations

import threading
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

_RESOLUTIONS = {
    "1080×1080": (1080, 1080),
    "1920×1080": (1920, 1080),
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

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # --- Toolbar ---
        toolbar = ctk.CTkFrame(self, corner_radius=0)
        toolbar.grid(row=0, column=0, sticky="ew", padx=0, pady=0)
        self._build_toolbar(toolbar)

        # --- Preview strip (scrollable) ---
        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(10, 0))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self._preview = PreviewStrip(
            preview_frame,
            on_select=self._on_thumbnail_select,
        )
        self._preview.grid(row=0, column=0, sticky="nsew")

        # Enlarged selected-frame view
        self._detail_label = ctk.CTkLabel(preview_frame, text="")
        self._detail_label.grid(row=1, column=0, pady=4)

        # --- Status bar ---
        status_bar = ctk.CTkFrame(self, corner_radius=0, height=48)
        status_bar.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        status_bar.grid_propagate(False)
        self._build_status_bar(status_bar)

    def _build_toolbar(self, parent: ctk.CTkFrame):
        # Row 0: folder picker
        # Row 1: settings + action buttons
        parent.grid_columnconfigure(1, weight=1)   # folder label stretches

        # --- Row 0: folder ---
        ctk.CTkButton(
            parent, text="Open Folder…", command=self._pick_folder, width=130
        ).grid(row=0, column=0, padx=(10, 6), pady=(8, 2), sticky="w")

        self._folder_label = ctk.CTkLabel(parent, text="No folder selected", anchor="w")
        self._folder_label.grid(row=0, column=1, padx=(0, 10), pady=(8, 2), sticky="ew",
                                columnspan=99)

        # --- Row 1: settings ---
        r1 = ctk.CTkFrame(parent, fg_color="transparent")
        r1.grid(row=1, column=0, columnspan=99, sticky="ew", padx=10, pady=(2, 8))

        col = 0

        ctk.CTkLabel(r1, text="Resolution:").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._res_var = ctk.StringVar(value="1080×1080")
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

        # Spacer
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

        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._aligned_frames.clear()
        self._preview.clear()
        self._set_status(f"Aligning {len(images)} photos…")
        self._progress.set(0)

        # Run alignment on a background thread to keep UI responsive
        threading.Thread(
            target=self._align_worker,
            args=(images, output_size),
            daemon=True,
        ).start()

    def _align_worker(self, images: list[Path], output_size: tuple[int, int]):
        if self._aligner:
            self._aligner.close()
        self._aligner = FaceAligner(output_size=output_size)

        skipped = 0

        # --- Phase 1: detect faces in every image ---
        # Collect (path, chosen_face) pairs; multi-face images pause for UI pick.
        pending: list[tuple[Path, object]] = []
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

        if not pending:
            self.after(0, self._align_done, [], skipped)
            return

        # --- Phase 2: compute a data-driven target eye size ---
        #
        # For each image, ask: "if this photo were shrunk to fit inside the
        # output frame without cropping, how many pixels wide would the
        # inter-eye distance be?"  The median of those values becomes the
        # shared target for all images.
        #
        # Why this works:
        #   • Per-image natural scale is still used (face sizes are consistent).
        #   • The target is smaller than the default 30%-of-width, so images
        #     are zoomed out — showing more context, with black fill where the
        #     original doesn't reach the edge.
        #   • Clamped to [10%, 25%] of output width so faces stay legible.
        output_w, _ = output_size
        fit_eye_pxs: list[float] = []
        for path, face in pending:
            try:
                from PIL import Image as _PIL, ImageOps as _IOps
                with _PIL.open(path) as _img:
                    _img = _IOps.exif_transpose(_img)
                    orig_w, orig_h = _img.size
                fit_eye_pxs.append(self._aligner.eye_px_at_fit(face, orig_w, orig_h))
            except Exception:
                pass

        if fit_eye_pxs:
            raw_target = float(np.median(fit_eye_pxs))
            # clamp: faces must be between 10% and 25% of output width
            target_eye_px = float(np.clip(raw_target, 0.10 * output_w, 0.25 * output_w))
        else:
            target_eye_px = 0.25 * output_w  # safe fallback

        pct = round(target_eye_px / output_w * 100, 1)
        self._set_status(f"Target face size: {pct}% of frame — aligning…")

        # --- Phase 3: warp every image with the shared target ---
        # Each image uses its own per-image scale so that the inter-eye
        # distance equals target_eye_px in every output frame.  Consistent
        # face size across frames is what makes the video look aligned.
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

        self.after(0, self._align_done, results, skipped)

    def _ask_face_pick(self, path: Path, faces) -> Optional[object]:
        """Block the worker thread until the user picks a face in the dialog."""
        result_holder = [None]
        event = threading.Event()

        def _show():
            result_holder[0] = pick_face(self, path, faces)
            event.set()

        self.after(0, _show)
        event.wait()
        return result_holder[0]

    def _align_done(self, results: list[AlignedFrame], skipped: int):
        self._aligned_frames = results
        self._preview.set_frames(results)
        self._align_btn.configure(state="normal")

        msg = f"Aligned {len(results)} photo(s)."
        if skipped:
            msg += f" {skipped} skipped."
        self._set_status(msg)
        self._progress.set(1.0)

        if results:
            self._export_btn.configure(state="normal")

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

    def _on_thumbnail_select(self, idx: int):
        frame = self._aligned_frames[idx]
        self._detail_label.configure(text=frame.source_path.name)

    # ------------------------------------------------------------------

    def _set_status(self, msg: str):
        self._status_label.configure(text=msg)

    def on_close(self):
        if self._aligner:
            self._aligner.close()
        self.destroy()
