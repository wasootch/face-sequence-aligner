"""Audio track selection and playback mixin."""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from tkinter import filedialog, messagebox
from typing import Optional

from ui.utils import _NO_AUDIO


class AudioMixin:
    """Audio track selection and ffplay preview.

    All attributes accessed via self are initialised in App.__init__.
    """

    def _browse_audio(self):
        path = filedialog.askopenfilename(
            title="Select audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.ogg *.flac *.m4a *.aac"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        p = Path(path)
        label = f"Custom: {p.name}"
        self._audio_tracks[label] = p
        self._audio_menu.configure(values=[_NO_AUDIO] + list(self._audio_tracks.keys()))
        self._audio_var.set(label)
        self._audio_custom_label.configure(text=str(p.parent))

    def _on_audio_track_changed(self):
        self._stop_audio()
        path = self._selected_audio_path
        if path and path.exists():
            self._audio_duration_label.configure(text=self._get_audio_duration(path))
            self._audio_play_btn.configure(state="normal")
        else:
            self._audio_duration_label.configure(text="")
            self._audio_play_btn.configure(state="disabled")

    def _toggle_audio_playback(self):
        if self._audio_player is not None:
            self._stop_audio()
        else:
            self._start_audio()

    def _start_audio(self):
        path = self._selected_audio_path
        if not path or not path.exists():
            return
        import shutil
        if not shutil.which("ffplay"):
            messagebox.showwarning(
                "ffplay not found",
                "Audio preview requires ffplay (installed with ffmpeg).",
            )
            return
        self._audio_player = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._audio_play_btn.configure(text="■  Stop")
        threading.Thread(
            target=self._monitor_playback,
            args=(self._audio_player,),
            daemon=True,
        ).start()

    def _stop_audio(self):
        if self._audio_player is not None:
            self._audio_player.terminate()
            self._audio_player = None
        self._audio_play_btn.configure(text="▶  Play")

    def _monitor_playback(self, proc: subprocess.Popen):
        proc.wait()
        self.after(0, self._on_playback_ended, proc)

    def _on_playback_ended(self, proc: subprocess.Popen):
        if self._audio_player is proc:
            self._audio_player = None
            self._audio_play_btn.configure(text="▶  Play")

    @staticmethod
    def _get_audio_duration(path: Path) -> str:
        import shutil
        if not shutil.which("ffprobe"):
            return ""
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "quiet",
                 "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1",
                 str(path)],
                capture_output=True, text=True, timeout=5,
            )
            secs = float(result.stdout.strip())
            m, s = divmod(int(secs), 60)
            return f"{m}:{s:02d}"
        except Exception:
            return ""

    @property
    def _selected_audio_path(self) -> Optional[Path]:
        return self._audio_tracks.get(self._audio_var.get())
