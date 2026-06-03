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
        parent.grid_columnconfigure(10, weight=1)  # spacer

        col = 0

        # Folder button
        ctk.CTkButton(
            parent, text="Open Folder…", command=self._pick_folder, width=130
        ).grid(row=0, column=col, padx=(10, 4), pady=8)
        col += 1

        self._folder_label = ctk.CTkLabel(parent, text="No folder selected", anchor="w")
        self._folder_label.grid(row=0, column=col, padx=(0, 16), pady=8, sticky="w")
        col += 1

        # Separator
        ctk.CTkLabel(parent, text="|", text_color="gray").grid(row=0, column=col, padx=4)
        col += 1

        # Resolution
        ctk.CTkLabel(parent, text="Resolution:").grid(row=0, column=col, padx=(4, 2))
        col += 1
        self._res_var = ctk.StringVar(value="1080×1080")
        ctk.CTkOptionMenu(
            parent, variable=self._res_var, values=list(_RESOLUTIONS.keys()), width=130
        ).grid(row=0, column=col, padx=(0, 12), pady=8)
        col += 1

        # Hold duration
        ctk.CTkLabel(parent, text="Hold (s):").grid(row=0, column=col, padx=(4, 2))
        col += 1
        self._hold_var = ctk.StringVar(value="1.5")
        ctk.CTkEntry(parent, textvariable=self._hold_var, width=55).grid(
            row=0, column=col, padx=(0, 12), pady=8
        )
        col += 1

        # Transition duration
        ctk.CTkLabel(parent, text="Transition (s):").grid(row=0, column=col, padx=(4, 2))
        col += 1
        self._trans_var = ctk.StringVar(value="1.0")
        ctk.CTkEntry(parent, textvariable=self._trans_var, width=55).grid(
            row=0, column=col, padx=(0, 12), pady=8
        )
        col += 1

        # FPS
        ctk.CTkLabel(parent, text="FPS:").grid(row=0, column=col, padx=(4, 2))
        col += 1
        self._fps_var = ctk.StringVar(value="30")
        ctk.CTkEntry(parent, textvariable=self._fps_var, width=45).grid(
            row=0, column=col, padx=(0, 12), pady=8
        )
        col += 1

        # Spacer
        ctk.CTkLabel(parent, text="").grid(row=0, column=col, sticky="ew")
        col += 1

        # Align button
        self._align_btn = ctk.CTkButton(
            parent, text="Align Photos", command=self._start_align, width=120
        )
        self._align_btn.grid(row=0, column=col, padx=(0, 8), pady=8)
        col += 1

        # Export button
        self._export_btn = ctk.CTkButton(
            parent,
            text="Export MP4…",
            command=self._start_export,
            width=120,
            state="disabled",
        )
        self._export_btn.grid(row=0, column=col, padx=(0, 10), pady=8)

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
        results: list[AlignedFrame] = []
        skipped = 0
        pending_pick: Optional[tuple] = None  # (image_path, faces) awaiting UI pick

        if self._aligner:
            self._aligner.close()
        self._aligner = FaceAligner(output_size=output_size)

        for i, path in enumerate(images):
            self._set_status(f"[{i+1}/{len(images)}] {path.name}")
            self._progress.set((i + 1) / len(images))

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
                # Must ask on main thread
                chosen = self._ask_face_pick(path, faces)
                if chosen is None:
                    skipped += 1
                    continue

            try:
                frame = self._aligner.align(path, chosen)
                results.append(frame)
            except Exception as exc:
                self._set_status(f"Align error {path.name}: {exc}")
                skipped += 1

        # Back on main thread
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
        self.after(0, self._progress.set, current / total)
        self.after(0, self._set_status, f"Exporting frame {current}/{total}…")

    def _export_done(self, error: Optional[Exception]):
        if error:
            self.after(0, messagebox.showerror, "Export failed", str(error))
            self.after(0, self._set_status, "Export failed.")
        else:
            self.after(0, self._set_status, "Export complete!")
            self.after(0, self._progress.set, 1.0)
        self.after(0, self._export_btn.configure, {"state": "normal"})
        self.after(0, self._align_btn.configure, {"state": "normal"})

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
