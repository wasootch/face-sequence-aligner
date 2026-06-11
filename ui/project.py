"""Project save / load mixin."""

from __future__ import annotations

import cv2
import json
import threading
import zipfile
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import numpy as np

from aligner import AlignedFrame, Face
from ui.utils import _NO_AUDIO, _RESOLUTIONS, _SORT_OPTIONS


class ProjectMixin:
    """Adds save/load/new-project methods to the App class.

    All attributes accessed via self are initialised in App.__init__.
    """

    # ------------------------------------------------------------------
    # Public actions (wired to menu / keyboard shortcuts)
    # ------------------------------------------------------------------

    def _new_project(self):
        self._project_path = None
        self._folder = None
        self._folder_label.configure(text="No folder selected")
        self._aligned_frames.clear()
        self._pending_faces.clear()
        self._detected_folder = None
        self._aligned_output_size = None
        self._face_pct_var.set("")
        self._preview.clear()
        self._last_export_path = None
        self._export_btn.configure(state="disabled")
        self._show_folder_btn.configure(state="disabled")
        self._video_info_label.configure(text="")
        self._set_status("Ready.")
        self._progress.set(0)
        self._update_title()

    def _open_project(self):
        path = filedialog.askopenfilename(
            title="Open Project",
            filetypes=[("Face Sequence Aligner project", "*.fsa"), ("All files", "*.*")],
        )
        if path:
            self._do_load(Path(path))

    def _save_project(self):
        if self._project_path:
            self._do_save(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        default_name = self._folder.name if self._folder else ""
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".fsa",
            filetypes=[("Face Sequence Aligner project", "*.fsa")],
            initialfile=default_name,
        )
        if path:
            self._project_path = Path(path)
            self._do_save(self._project_path)

    def _update_title(self):
        if self._project_path:
            self.title(f"Face Sequence Aligner — {self._project_path.stem}")
        else:
            self.title("Face Sequence Aligner")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def _do_save(self, path: Path):
        audio_path = self._audio_tracks.get(self._audio_var.get())

        face_entries = [
            {
                "path":        str(af.source_path),
                "frame_index": i,
                "face_count":  af.face_count,
                "face": {
                    "index":     af.face.index,
                    "bbox":      list(af.face.bbox),
                    "left_eye":  list(af.face.left_eye),
                    "right_eye": list(af.face.right_eye),
                },
            }
            for i, af in enumerate(self._aligned_frames)
        ]

        data = {
            "version": 2,
            "folder":  str(self._folder) if self._folder else None,
            "settings": {
                "resolution": self._res_var.get(),
                "hold":       self._hold_var.get(),
                "transition": self._trans_var.get(),
                "fps":        self._fps_var.get(),
                "face_pct":   self._face_pct_var.get(),
                "sort":       self._sort_var.get(),
                "audio_path": str(audio_path) if audio_path else None,
            },
            "faces": face_entries,
        }

        frames_snapshot = list(self._aligned_frames)

        self._set_status("Saving project…")
        self._progress.configure(mode="indeterminate")
        self._progress.start()

        def _worker():
            try:
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                    zf.writestr("project.json", json.dumps(data, indent=2))
                    for i, frame in enumerate(frames_snapshot):
                        ok, buf = cv2.imencode(
                            ".jpg", frame.image,
                            [cv2.IMWRITE_JPEG_QUALITY, 92],
                        )
                        if ok:
                            zf.writestr(f"frames/{i:04d}.jpg", buf.tobytes())

                def _done():
                    self._progress.stop()
                    self._progress.configure(mode="determinate")
                    self._progress.set(1.0)
                    self._update_title()
                    self._set_status(f"Project saved: {path.name}")
                self.after(0, _done)

            except Exception as exc:
                def _err(e=exc):
                    self._progress.stop()
                    self._progress.configure(mode="determinate")
                    messagebox.showerror("Save failed", str(e))
                    self._set_status("Save failed.")
                self.after(0, _err)

        threading.Thread(target=_worker, daemon=True).start()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def _do_load(self, path: Path):
        is_zip = zipfile.is_zipfile(str(path))
        try:
            if is_zip:
                with zipfile.ZipFile(path, "r") as zf:
                    data = json.loads(zf.read("project.json").decode("utf-8"))
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not read project file:\n{exc}")
            return

        self._project_path = path

        folder_str = data.get("folder")
        self._folder = Path(folder_str) if folder_str else None
        self._folder_label.configure(
            text=str(self._folder) if self._folder else "No folder selected"
        )

        s = data.get("settings", {})
        if s.get("resolution") in _RESOLUTIONS:
            self._res_var.set(s["resolution"])
        for var, key in (
            (self._hold_var,     "hold"),
            (self._trans_var,    "transition"),
            (self._fps_var,      "fps"),
            (self._face_pct_var, "face_pct"),
        ):
            if key in s:
                var.set(s[key])
        if s.get("sort") in _SORT_OPTIONS:
            self._sort_var.set(s["sort"])

        audio_path_str = s.get("audio_path")
        if audio_path_str:
            ap = Path(audio_path_str)
            if ap.exists():
                label = ap.name
                self._audio_tracks[label] = ap
                self._audio_menu.configure(values=[_NO_AUDIO] + list(self._audio_tracks.keys()))
                self._audio_var.set(label)
            else:
                self._audio_var.set(_NO_AUDIO)
        else:
            self._audio_var.set(_NO_AUDIO)

        # (path, face, frame_index, face_count)
        pending_all: list[tuple[Path, Face, int, int]] = []
        for i, entry in enumerate(data.get("faces", [])):
            fd = entry.get("face", {})
            try:
                face = Face(
                    index=fd["index"],
                    bbox=tuple(fd["bbox"]),
                    left_eye=tuple(fd["left_eye"]),
                    right_eye=tuple(fd["right_eye"]),
                )
            except (KeyError, TypeError):
                continue
            pending_all.append((
                Path(entry["path"]),
                face,
                entry.get("frame_index", i),
                entry.get("face_count", 1),
            ))

        if not pending_all:
            messagebox.showwarning("No photos loaded", "The project contains no photo entries.")
            self._update_title()
            return

        self._update_title()

        if is_zip and data.get("version", 1) >= 2:
            with zipfile.ZipFile(path, "r") as zf:
                zip_names = set(zf.namelist())
            has_frames = any(
                f"frames/{fi:04d}.jpg" in zip_names for _, _, fi, _ in pending_all
            )
            if has_frames:
                output_size = _RESOLUTIONS.get(self._res_var.get(), (1080, 1080))
                try:
                    used_pct: Optional[float] = float(self._face_pct_var.get())
                except ValueError:
                    used_pct = None
                self._start_load_from_zip(path, pending_all, output_size, used_pct)
                return

        pending_existing = [(p, f) for p, f, _, _ in pending_all if p.exists()]
        if not pending_existing:
            messagebox.showwarning(
                "No photos found",
                "None of the saved photos could be found on disk.\n"
                "Check that the photo folder is still accessible.",
            )
            return
        self._start_align_from_project(pending_existing)

    def _start_align_from_project(self, pending: list[tuple[Path, Face]]):
        """Re-align from saved face data (no detection, no re-sort)."""
        output_size = _RESOLUTIONS.get(self._res_var.get(), (1080, 1080))

        self._pending_faces = list(pending)
        self._detected_folder = self._folder

        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._aligned_frames.clear()
        self._preview.clear()
        self._progress.configure(mode="indeterminate")
        self._progress.start()
        self._set_status(f"Loading project — aligning {len(pending)} photo(s)…")

        threading.Thread(
            target=self._align_worker,
            args=([], output_size, True, None),  # sort_option=None → preserve order
            daemon=True,
        ).start()

    def _start_load_from_zip(
        self,
        zip_path: Path,
        pending_all: list[tuple[Path, Face, int, int]],
        output_size: tuple[int, int],
        used_face_pct: Optional[float],
    ):
        """Load pre-aligned frames from the ZIP archive on a background thread."""
        self._pending_faces = [(p, f) for p, f, _, _ in pending_all]
        self._detected_folder = self._folder

        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._aligned_frames.clear()
        self._preview.clear()
        self._progress.configure(mode="indeterminate")
        self._progress.start()
        self._set_status(f"Loading project — reading {len(pending_all)} frame(s)…")

        threading.Thread(
            target=self._load_zip_worker,
            args=(zip_path, pending_all, output_size, used_face_pct),
            daemon=True,
        ).start()

    def _load_zip_worker(
        self,
        zip_path: Path,
        pending_all: list[tuple[Path, Face, int, int]],
        output_size: tuple[int, int],
        used_face_pct: Optional[float],
    ):
        try:
            frames: list[AlignedFrame] = []
            skipped = 0
            n = len(pending_all)

            with zipfile.ZipFile(zip_path, "r") as zf:
                zip_names = set(zf.namelist())
                for i, (img_path, face, frame_index, face_count) in enumerate(pending_all):
                    self._set_status(f"Loading frame {i + 1}/{n}…")
                    frame_name = f"frames/{frame_index:04d}.jpg"
                    if frame_name not in zip_names:
                        skipped += 1
                        continue
                    buf = np.frombuffer(zf.read(frame_name), dtype=np.uint8)
                    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                    if bgr is None:
                        skipped += 1
                        continue
                    frames.append(AlignedFrame(
                        source_path=img_path,
                        image=bgr,
                        face=face,
                        transform=np.zeros((2, 3), dtype=np.float64),
                        face_count=face_count,
                    ))

            self.after(0, lambda: self._align_done(frames, skipped, used_face_pct, output_size))

        except Exception as exc:
            def _err(e=exc):
                self._progress.stop()
                self._progress.configure(mode="determinate")
                self._align_btn.configure(state="normal")
                messagebox.showerror("Load failed", str(e))
                self._set_status("Load failed.")
            self.after(0, _err)
