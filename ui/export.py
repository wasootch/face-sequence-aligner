"""Export pipeline and output panel mixin."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from exporter import export_mp4_async


class ExportMixin:
    """MP4 export and output-panel methods.

    All attributes accessed via self are initialised in App.__init__.
    """

    def _start_export(self):
        if not self._aligned_frames:
            return

        if self._last_export_path:
            default_name = self._last_export_path.stem
            initial_dir  = str(self._last_export_path.parent)
        elif self._project_path:
            default_name = self._project_path.stem
            initial_dir  = str(self._project_path.parent)
        else:
            default_name = ""
            initial_dir  = ""

        out_path = filedialog.asksaveasfilename(
            title="Save MP4",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4")],
            initialfile=default_name,
            initialdir=initial_dir or None,
        )
        if not out_path:
            return
        self._last_export_path = Path(out_path)

        try:
            fps  = int(self._fps_var.get())
            hold = float(self._hold_var.get())
            trans = float(self._trans_var.get())
        except ValueError:
            messagebox.showerror("Invalid settings", "FPS, hold, and transition must be numbers.")
            return

        self._export_btn.configure(state="disabled")
        self._align_btn.configure(state="disabled")
        self._set_status("Exporting…")
        self._progress.set(0)

        audio = self._selected_audio_path
        if audio and not __import__("shutil").which("ffmpeg"):
            messagebox.showwarning(
                "ffmpeg not found",
                "Audio mixing requires ffmpeg on your PATH.\n"
                "The video will be exported without audio.",
            )
            audio = None

        self._export_thread = export_mp4_async(
            frames=self._aligned_frames,
            output_path=Path(out_path),
            fps=fps,
            hold_seconds=hold,
            transition_seconds=trans,
            progress_callback=self._export_progress,
            done_callback=self._export_done,
            audio_path=audio,
            status_callback=self._export_status,
        )

    def _export_progress(self, current: int, total: int):
        msg = f"Exporting frame {current}/{total}…"
        def _update():
            if current == total:
                self._progress.configure(mode="indeterminate")
                self._progress.start()
            else:
                self._progress.set(current / total)
            self._set_status(msg)
        self.after(0, _update)

    def _export_status(self, msg: str):
        self.after(0, lambda: self._set_status(msg))

    def _export_done(self, error: Optional[Exception]):
        def _on_main_thread():
            self._progress.stop()
            self._progress.configure(mode="determinate")
            if error:
                messagebox.showerror("Export failed", str(error))
                self._set_status("Export failed.")
                self._progress.set(0)
            else:
                self._set_status("Export complete!")
                self._progress.set(1.0)
                self._show_folder_btn.configure(state="normal")
            self._export_btn.configure(state="normal")
            self._align_btn.configure(state="normal")
        self.after(0, _on_main_thread)

    def _compute_duration(self) -> str:
        n = len(self._aligned_frames)
        if n == 0:
            return ""
        try:
            fps  = int(self._fps_var.get())
            hold = float(self._hold_var.get())
            trans = float(self._trans_var.get())
        except ValueError:
            return ""
        if fps <= 0:
            return ""
        total = n * hold + max(0, n - 1) * trans
        m, s = divmod(int(total), 60)
        return f"{n} photo{'s' if n != 1 else ''} · {m}:{s:02d}"

    def _update_video_info(self, *_):
        if hasattr(self, "_video_info_label"):
            self._video_info_label.configure(text=self._compute_duration())

    def _show_in_folder(self):
        if self._last_export_path and self._last_export_path.exists():
            subprocess.Popen(["explorer", "/select,", str(self._last_export_path)])
