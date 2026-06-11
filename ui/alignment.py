"""Face detection, alignment pipeline, and thumbnail interaction mixin."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

import cv2
import numpy as np

from aligner import AlignedFrame, Face, FaceAligner
from ui.face_picker import pick_face
from ui.menu import _PopupMenu
from ui.utils import _IMAGE_EXTS, _RESOLUTIONS, _sort_images_keyed


class AlignMixin:
    """Folder picking, face detection/alignment, and thumbnail management.

    All attributes accessed via self are initialised in App.__init__.
    """

    # ------------------------------------------------------------------
    # Folder selection
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
        self._pending_faces.clear()
        self._detected_folder = None
        self._face_pct_var.set("")

        images = sorted(
            p for p in self._folder.iterdir()
            if p.suffix.lower() in _IMAGE_EXTS
        )
        if not images:
            self._set_status("No images found in folder.")
            return

        sorted_images = [
            p for p, _ in _sort_images_keyed([(p, None) for p in images], self._sort_var.get())
        ]
        token = object()
        self._raw_preview_token = token
        self._progress.configure(mode="indeterminate")
        self._progress.start()
        self._set_status(f"Loading previews for {len(sorted_images)} photo(s)…")
        threading.Thread(
            target=self._load_raw_previews,
            args=(sorted_images, token),
            daemon=True,
        ).start()

    def _load_raw_previews(self, images: list[Path], token: object) -> None:
        from PIL import Image as _PIL, ImageOps as _IOps

        dummy_face = Face(index=0, bbox=(0, 0, 0, 0), left_eye=(0.0, 0.0), right_eye=(0.0, 0.0))
        frames: list[AlignedFrame] = []

        for path in images:
            if token is not self._raw_preview_token:
                return
            try:
                pil = _IOps.exif_transpose(_PIL.open(path).convert("RGB"))
                pil.thumbnail((240, 240), _PIL.LANCZOS)
                bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                frames.append(AlignedFrame(
                    source_path=path,
                    image=bgr,
                    face=dummy_face,
                    transform=np.zeros((2, 3), dtype=np.float64),
                    face_count=0,
                ))
            except Exception:
                continue

        def _done(tok=token, fs=frames):
            if tok is not self._raw_preview_token:
                return
            self._progress.stop()
            self._progress.configure(mode="determinate")
            self._progress.set(0)
            self._preview.set_frames(fs)
            self._set_status(f"{len(fs)} photo(s) ready. Click 'Align Photos' to process.")

        self.after(0, _done)

    # ------------------------------------------------------------------
    # Alignment pipeline
    # ------------------------------------------------------------------

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

        output_size = _RESOLUTIONS.get(self._res_var.get(), (1080, 1080))
        can_skip = bool(self._pending_faces) and self._detected_folder == self._folder

        self._raw_preview_token = object()
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._aligned_frames.clear()
        self._progress.set(0)

        if can_skip:
            self._set_status("Re-aligning with new face size (skipping detection)…")
        else:
            self._set_status(f"Detecting faces in {len(images)} photo(s)…")

        threading.Thread(
            target=self._align_worker,
            args=(images, output_size, can_skip, self._sort_var.get()),
            daemon=True,
        ).start()

    def _align_worker(
        self,
        images: list[Path],
        output_size: tuple[int, int],
        skip_detection: bool,
        sort_option: Optional[str],
    ):
        if self._aligner:
            self._aligner.close()
        self._aligner = FaceAligner(output_size=output_size)

        skipped = 0
        face_counts: dict[Path, int] = {}

        # Phase 1: detect
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
                face_counts[path] = len(faces)
                pending.append((path, chosen))

            self._pending_faces = pending
            self._detected_folder = self._folder

        if not pending:
            self.after(0, self._align_done, [], skipped, None)
            return

        if sort_option is not None:
            pending = _sort_images_keyed(pending, sort_option)

        # Phase 2: compute target eye size
        output_w, _ = output_size
        target_eye_px: Optional[float] = None
        try:
            user_pct = float(self._face_pct_var.get())
            if 1.0 <= user_pct <= 80.0:
                target_eye_px = user_pct / 100.0 * output_w
        except ValueError:
            pass

        if target_eye_px is None:
            from PIL import Image as _PIL, ImageOps as _IOps
            fit_eye_pxs: list[float] = []
            for path, face in pending:
                try:
                    with _PIL.open(path) as img:
                        img = _IOps.exif_transpose(img)
                        orig_w, orig_h = img.size
                    fit_eye_pxs.append(self._aligner.eye_px_at_fit(face, orig_w, orig_h))
                except Exception:
                    pass
            if fit_eye_pxs:
                raw = float(np.median(fit_eye_pxs))
                target_eye_px = float(np.clip(raw, 0.05 * output_w, 0.25 * output_w))
            else:
                target_eye_px = 0.20 * output_w

        # Phase 3: warp
        results: list[AlignedFrame] = []
        m = len(pending)
        for j, (path, face) in enumerate(pending):
            self._set_status(f"Aligning [{j+1}/{m}] {path.name}")
            self._progress.set(0.5 + (j + 1) / (m * 2))
            try:
                frame = self._aligner.align(path, face, target_eye_px=target_eye_px)
                frame.face_count = face_counts.get(path, 1)
                results.append(frame)
            except Exception as exc:
                self._set_status(f"Align error {path.name}: {exc}")
                skipped += 1

        self.after(0, self._align_done, results, skipped,
                   target_eye_px / output_w * 100, output_size)

    def _on_sort_changed(self) -> None:
        if not self._pending_faces or not self._aligned_frames:
            return
        self._pending_faces = _sort_images_keyed(self._pending_faces, self._sort_var.get())
        path_to_frame = {f.source_path: f for f in self._aligned_frames}
        self._aligned_frames = [
            path_to_frame[p] for p, _ in self._pending_faces if p in path_to_frame
        ]
        self._preview.set_frames(self._aligned_frames)
        self._update_video_info()

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
        self._progress.stop()
        self._progress.configure(mode="determinate")
        self._aligned_frames = results
        self._aligned_output_size = output_size
        self._preview.set_frames(results)
        self._align_btn.configure(state="normal")

        if used_face_pct is not None:
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
        self._update_video_info()

    # ------------------------------------------------------------------
    # Thumbnail interactions
    # ------------------------------------------------------------------

    def _on_thumbnail_select(self, idx: int):
        if 0 <= idx < len(self._aligned_frames):
            self._detail_label.configure(text=self._aligned_frames[idx].source_path.name)

    def _on_thumbnail_reorder(self, old_idx: int, new_idx: int):
        frame = self._aligned_frames.pop(old_idx)
        self._aligned_frames.insert(new_idx, frame)

        if len(self._pending_faces) == len(self._aligned_frames):
            path_to_pair = {p: f for p, f in self._pending_faces}
            self._pending_faces = [
                (af.source_path, path_to_pair[af.source_path])
                for af in self._aligned_frames
                if af.source_path in path_to_pair
            ]

        self._preview.set_frames(self._aligned_frames)

    def _on_thumbnail_context(self, idx: int, x: int, y: int):
        menu = _PopupMenu(self)
        menu.add_command("Remove from sequence",  lambda: self._remove_frame(idx))
        menu.add_command("Choose different face…", lambda: self._repick_face(idx))
        menu.show(x, y)

    def _remove_frame(self, idx: int):
        if not (0 <= idx < len(self._aligned_frames)):
            return
        name = self._aligned_frames[idx].source_path.name
        self._aligned_frames.pop(idx)
        if 0 <= idx < len(self._pending_faces):
            self._pending_faces.pop(idx)
        self._preview.set_frames(self._aligned_frames)
        if not self._aligned_frames:
            self._export_btn.configure(state="disabled")
        self._set_status(f"Removed {name}. {len(self._aligned_frames)} photo(s) remaining.")

    def _repick_face(self, idx: int):
        if not (0 <= idx < len(self._aligned_frames)):
            return
        path = self._aligned_frames[idx].source_path

        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._set_status(f"Detecting faces in {path.name}…")

        def _worker():
            try:
                if self._aligner is None:
                    self._aligner = FaceAligner(
                        output_size=_RESOLUTIONS.get(self._res_var.get(), (1080, 1080))
                    )

                faces = self._aligner.detect_faces(path)

                if not faces:
                    def _no_face():
                        messagebox.showwarning("No face found", f"No face detected in {path.name}.")
                        self._align_btn.configure(state="normal")
                        self._export_btn.configure(state="normal")
                        self._set_status("Ready.")
                    self.after(0, _no_face)
                    return

                chosen = faces[0] if len(faces) == 1 else self._ask_face_pick(path, faces)
                if chosen is None:
                    def _cancelled():
                        self._align_btn.configure(state="normal")
                        self._export_btn.configure(state="normal")
                        self._set_status("Re-pick cancelled.")
                    self.after(0, _cancelled)
                    return

                output_size = self._aligned_output_size or _RESOLUTIONS.get(
                    self._res_var.get(), (1080, 1080)
                )
                try:
                    target_eye_px = float(self._face_pct_var.get()) / 100.0 * output_size[0]
                except ValueError:
                    target_eye_px = 0.20 * output_size[0]

                frame = self._aligner.align(path, chosen, target_eye_px=target_eye_px)
                frame.face_count = len(faces)

                def _done():
                    self._aligned_frames[idx] = frame
                    if 0 <= idx < len(self._pending_faces):
                        self._pending_faces[idx] = (path, chosen)
                    self._preview.set_frames(self._aligned_frames)
                    self._align_btn.configure(state="normal")
                    self._export_btn.configure(state="normal")
                    self._set_status(f"Re-aligned {path.name}.")
                self.after(0, _done)

            except Exception as exc:
                def _err(e=exc):
                    messagebox.showerror("Re-pick failed", str(e))
                    self._align_btn.configure(state="normal")
                    self._export_btn.configure(state="normal")
                    self._set_status("Re-pick failed.")
                self.after(0, _err)

        threading.Thread(target=_worker, daemon=True).start()
