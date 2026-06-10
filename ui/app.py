"""
Main application window.

Layout (top → bottom):
  Toolbar   — folder picker (row 0), settings + action buttons (row 1)
  Preview   — aligned-frame thumbnail strip
  Status    — progress bar + status label
"""

from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox
import tkinter as tk
from typing import Optional

import numpy as np
import customtkinter as ctk

from aligner import AlignedFrame, Face, FaceAligner
try:
    from version import __version__ as _APP_VERSION
except ImportError:
    _APP_VERSION = "unknown"
from exporter import export_mp4_async
from ui.preview import PreviewStrip
from ui.face_picker import pick_face

_IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
_AUDIO_EXTS  = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac"}
_MUSIC_DIR   = Path(__file__).parent.parent / "music"
_NO_AUDIO    = "None"


def _scan_music() -> dict[str, Path]:
    """Return {filename: path} for audio files in the music/ folder, sorted by name."""
    if not _MUSIC_DIR.exists():
        return {}
    return {
        p.name: p
        for p in sorted(_MUSIC_DIR.iterdir())
        if p.suffix.lower() in _AUDIO_EXTS
    }

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


class _PopupMenu:
    """
    Custom styled popup menu that matches the app's dark theme.
    tk.Menu ignores bg/fg on Windows 11, so we build our own.
    """
    _W          = 220   # fixed menu width
    _BG         = "#2b2b2b"
    _BORDER     = "#555555"
    _FG         = "#dce4ee"
    _FG_DIM     = "#666666"
    _HOVER_BG   = "#1f538d"
    _SEP_COLOR  = "#444444"

    def __init__(self, parent: ctk.CTk):
        self._parent = parent
        self._items: list[tuple] = []

    def add_command(self, label: str, command, accelerator: str = ""):
        self._items.append(("cmd", label, command, accelerator))

    def add_separator(self):
        self._items.append(("sep",))

    def show(self, x: int, y: int):
        popup = tk.Toplevel(self._parent)
        popup.withdraw()
        popup.overrideredirect(True)
        popup.configure(bg=self._BORDER)

        inner = tk.Frame(popup, bg=self._BG)
        inner.pack(fill="both", expand=True, padx=1, pady=1)

        closed = [False]

        def close(cmd=None):
            if closed[0]:
                return
            closed[0] = True
            try:
                popup.destroy()
            except tk.TclError:
                pass
            if cmd:
                self._parent.after(0, cmd)

        for item in self._items:
            if item[0] == "sep":
                tk.Frame(inner, height=1, bg=self._SEP_COLOR).pack(
                    fill="x", padx=8, pady=3
                )
            else:
                _, label, command, accel = item
                row = tk.Frame(inner, bg=self._BG, cursor="hand2")
                row.pack(fill="x")
                lbl = tk.Label(
                    row, text=label, bg=self._BG, fg=self._FG,
                    font=("Segoe UI", 13), anchor="w", padx=12, pady=5,
                )
                lbl.pack(side="left", fill="both", expand=True)
                widgets = [row, lbl]
                if accel:
                    ak = tk.Label(
                        row, text=accel, bg=self._BG, fg=self._FG_DIM,
                        font=("Segoe UI", 13), anchor="e", padx=12,
                    )
                    ak.pack(side="right")
                    widgets.append(ak)

                def _bind(ws, cmd):
                    bg, hbg = self._BG, self._HOVER_BG
                    for w in ws:
                        w.bind("<Enter>", lambda e, _ws=ws: [_w.configure(bg=hbg) for _w in _ws])
                        w.bind("<Leave>", lambda e, _ws=ws: [_w.configure(bg=bg)  for _w in _ws])
                        w.bind("<Button-1>", lambda e, _c=cmd: close(_c))

                _bind(widgets, command)

        # Delay FocusOut binding so the popup has time to fully appear
        popup.after(150, lambda: popup.bind("<FocusOut>", lambda e: close()))
        popup.bind("<Escape>", lambda e: close())

        popup.update_idletasks()
        h = popup.winfo_reqheight()
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        popup.geometry(f"{self._W}x{h}+{min(x, sw - self._W)}+{min(y, sh - h)}")
        popup.deiconify()
        popup.lift()
        popup.focus_force()


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

        # Audio: maps dropdown label → Path; None means no audio.
        self._audio_tracks: dict[str, Path] = {}
        self._audio_player: Optional["subprocess.Popen"] = None

        self._project_path: Optional[Path] = None

        self._build_menu()
        self._build_ui()
        self.bind("<Control-n>", lambda _: self._new_project())
        self.bind("<Control-o>", lambda _: self._open_project())
        self.bind("<Control-s>", lambda _: self._save_project())

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------

    def _build_menu(self):
        bar = ctk.CTkFrame(self, corner_radius=0, height=30, fg_color=("#e0e0e0", "#1a1a1a"))
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_propagate(False)

        file_menu = _PopupMenu(self)
        file_menu.add_command("New",              self._new_project,     "Ctrl+N")
        file_menu.add_separator()
        file_menu.add_command("Open Project…",    self._open_project,    "Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command("Save Project",     self._save_project,    "Ctrl+S")
        file_menu.add_command("Save Project As…", self._save_project_as)
        file_menu.add_separator()
        file_menu.add_command("Exit", self.on_close)

        help_menu = _PopupMenu(self)
        help_menu.add_command("About", self._show_about)

        for label, menu in (("File", file_menu), ("Help", help_menu)):
            self._add_menu_button(bar, label, menu)

    def _add_menu_button(self, bar: ctk.CTkFrame, label: str, menu: _PopupMenu):
        btn = ctk.CTkButton(
            bar, text=label, width=52, height=26,
            fg_color="transparent",
            hover_color=("#cccccc", "#2d2d2d"),
            text_color=("black", "#dce4ee"),
            corner_radius=0,
        )
        btn.configure(command=lambda b=btn, m=menu: m.show(
            b.winfo_rootx(), b.winfo_rooty() + b.winfo_height()
        ))
        btn.pack(side="left", padx=0, pady=2)

    def _show_about(self):
        dlg = ctk.CTkToplevel(self)
        dlg.title("About Face Sequence Aligner")
        dlg.resizable(False, False)
        dlg.attributes("-toolwindow", True)  # removes minimize/maximize buttons on Windows

        dlg.withdraw()
        dlg.update_idletasks()
        w, h = 340, 200
        x = self.winfo_rootx() + (self.winfo_width()  - w) // 2
        y = self.winfo_rooty() + (self.winfo_height() - h) // 2
        dlg.geometry(f"{w}x{h}+{x}+{y}")
        dlg.deiconify()

        ctk.CTkLabel(
            dlg, text="Face Sequence Aligner",
            font=ctk.CTkFont(size=18, weight="bold"),
        ).pack(pady=(24, 4))
        ctk.CTkLabel(dlg, text=f"Version {_APP_VERSION}").pack()
        ctk.CTkLabel(
            dlg,
            text="Creates smooth face-aligned timelapse videos\nfrom a sequence of photos.",
            wraplength=300,
        ).pack(pady=(10, 0))
        ctk.CTkButton(dlg, text="OK", command=dlg.destroy, width=80).pack(pady=(18, 0))

        dlg.grab_set()
        dlg.wait_window()

    # ------------------------------------------------------------------
    # Project save / load
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
        self._export_btn.configure(state="disabled")
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
        path = filedialog.asksaveasfilename(
            title="Save Project",
            defaultextension=".fsa",
            filetypes=[("Face Sequence Aligner project", "*.fsa")],
        )
        if path:
            self._project_path = Path(path)
            self._do_save(self._project_path)

    def _do_save(self, path: Path):
        audio_choice = self._audio_var.get()
        audio_path = self._audio_tracks.get(audio_choice)

        data = {
            "version": 1,
            "folder": str(self._folder) if self._folder else None,
            "settings": {
                "resolution": self._res_var.get(),
                "hold":       self._hold_var.get(),
                "transition": self._trans_var.get(),
                "fps":        self._fps_var.get(),
                "face_pct":   self._face_pct_var.get(),
                "sort":       self._sort_var.get(),
                "audio_path": str(audio_path) if audio_path else None,
            },
            "faces": [
                {
                    "path": str(p),
                    "face": {
                        "index":     f.index,
                        "bbox":      list(f.bbox),
                        "left_eye":  list(f.left_eye),
                        "right_eye": list(f.right_eye),
                    },
                }
                for p, f in self._pending_faces
            ],
        }
        try:
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self._update_title()
            self._set_status(f"Project saved: {path.name}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _do_load(self, path: Path):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not read project file:\n{exc}")
            return

        if data.get("version", 1) != 1:
            messagebox.showwarning("Version mismatch", "This project was saved by a newer version of the app.")

        self._project_path = path

        # Restore folder
        folder_str = data.get("folder")
        self._folder = Path(folder_str) if folder_str else None
        self._folder_label.configure(
            text=str(self._folder) if self._folder else "No folder selected"
        )

        # Restore settings
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

        # Restore audio
        audio_path_str = s.get("audio_path")
        if audio_path_str:
            ap = Path(audio_path_str)
            if ap.exists():
                label = ap.name
                self._audio_tracks[label] = ap
                options = [_NO_AUDIO] + list(self._audio_tracks.keys())
                self._audio_menu.configure(values=options)
                self._audio_var.set(label)
            else:
                self._audio_var.set(_NO_AUDIO)
        else:
            self._audio_var.set(_NO_AUDIO)

        # Restore face detections
        pending: list[tuple[Path, Face]] = []
        for entry in data.get("faces", []):
            img_path = Path(entry["path"])
            if not img_path.exists():
                continue
            fd = entry.get("face", {})
            try:
                face = Face(
                    index=fd["index"],
                    bbox=tuple(fd["bbox"]),
                    left_eye=tuple(fd["left_eye"]),
                    right_eye=tuple(fd["right_eye"]),
                )
                pending.append((img_path, face))
            except (KeyError, TypeError):
                continue

        if not pending:
            messagebox.showwarning(
                "No photos loaded",
                "None of the saved photos could be found. Check that the photo folder is accessible.",
            )
            self._update_title()
            return

        self._update_title()
        self._start_align_from_project(pending)

    def _start_align_from_project(self, pending: list[tuple[Path, Face]]):
        """Align using pre-loaded face data (no detection, no re-sort)."""
        res_key = self._res_var.get()
        output_size = _RESOLUTIONS.get(res_key, (1080, 1080))

        self._pending_faces = list(pending)
        self._detected_folder = self._folder

        self._align_btn.configure(state="disabled")
        self._export_btn.configure(state="disabled")
        self._aligned_frames.clear()
        self._preview.clear()
        self._progress.set(0)
        self._set_status(f"Loading project — aligning {len(pending)} photo(s)…")

        threading.Thread(
            target=self._align_worker,
            args=([], output_size, True, None),   # sort_option=None → preserve order
            daemon=True,
        ).start()

    def _update_title(self):
        if self._project_path:
            self.title(f"Face Sequence Aligner — {self._project_path.stem}")
        else:
            self.title("Face Sequence Aligner")

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(self, corner_radius=0)
        toolbar.grid(row=1, column=0, sticky="ew", padx=0, pady=0)
        self._build_toolbar(toolbar)

        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(10, 0))
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)

        self._preview = PreviewStrip(
            preview_frame,
            on_select=self._on_thumbnail_select,
            on_reorder=self._on_thumbnail_reorder,
            on_context=self._on_thumbnail_context,
        )
        self._preview.grid(row=0, column=0, sticky="nsew")

        self._detail_label = ctk.CTkLabel(preview_frame, text="")
        self._detail_label.grid(row=1, column=0, pady=4)

        status_bar = ctk.CTkFrame(self, corner_radius=0, height=48)
        status_bar.grid(row=3, column=0, sticky="ew", padx=0, pady=0)
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
        self._res_var = ctk.StringVar(value="1920×1080")
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

        # --- Row 2: audio (separate frame so its columns don't affect row 1) ---
        r2 = ctk.CTkFrame(parent, fg_color="transparent")
        r2.grid(row=2, column=0, columnspan=99, sticky="w", padx=10, pady=(0, 8))

        ctk.CTkLabel(r2, text="Audio:").grid(row=0, column=0, padx=(0, 4))

        self._audio_tracks = _scan_music()
        audio_options = [_NO_AUDIO] + list(self._audio_tracks.keys())
        self._audio_var = ctk.StringVar(value=_NO_AUDIO)
        self._audio_var.trace_add("write", lambda *_: self._on_audio_track_changed())
        self._audio_menu = ctk.CTkOptionMenu(
            r2, variable=self._audio_var, values=audio_options, width=220,
        )
        self._audio_menu.grid(row=0, column=1, padx=(0, 8))

        self._audio_duration_label = ctk.CTkLabel(r2, text="", width=44, anchor="w", text_color="gray")
        self._audio_duration_label.grid(row=0, column=2, padx=(0, 6))

        self._audio_play_btn = ctk.CTkButton(
            r2, text="▶  Play", command=self._toggle_audio_playback, width=85, state="disabled"
        )
        self._audio_play_btn.grid(row=0, column=3, padx=(0, 8))

        ctk.CTkButton(
            r2, text="Browse…", command=self._browse_audio, width=80
        ).grid(row=0, column=4, padx=(0, 10))

        self._audio_custom_label = ctk.CTkLabel(r2, text="", anchor="w", text_color="gray")
        self._audio_custom_label.grid(row=0, column=5, sticky="w")

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
        sort_option: Optional[str],
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
        # sort_option=None means preserve the existing order (e.g. project load).
        if sort_option is not None:
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

    def _on_thumbnail_context(self, idx: int, x: int, y: int):
        menu = _PopupMenu(self)
        menu.add_command("Remove from sequence", lambda: self._remove_frame(idx))
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
                    output_size = _RESOLUTIONS.get(self._res_var.get(), (1080, 1080))
                    self._aligner = FaceAligner(output_size=output_size)

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

                output_size = self._aligned_output_size or _RESOLUTIONS.get(self._res_var.get(), (1080, 1080))
                try:
                    target_eye_px = float(self._face_pct_var.get()) / 100.0 * output_size[0]
                except ValueError:
                    target_eye_px = 0.20 * output_size[0]

                frame = self._aligner.align(path, chosen, target_eye_px=target_eye_px)

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
        options = [_NO_AUDIO] + list(self._audio_tracks.keys())
        self._audio_menu.configure(values=options)
        self._audio_var.set(label)           # triggers _on_audio_track_changed
        self._audio_custom_label.configure(text=str(p.parent))

    def _on_audio_track_changed(self):
        self._stop_audio()
        path = self._selected_audio_path
        if path and path.exists():
            dur = self._get_audio_duration(path)
            self._audio_duration_label.configure(text=dur)
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
            messagebox.showwarning("ffplay not found",
                                   "Audio preview requires ffplay (installed with ffmpeg).")
            return
        self._audio_player = subprocess.Popen(
            ["ffplay", "-nodisp", "-autoexit", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._audio_play_btn.configure(text="■  Stop")
        threading.Thread(target=self._monitor_playback,
                         args=(self._audio_player,), daemon=True).start()

    def _stop_audio(self):
        if self._audio_player is not None:
            self._audio_player.terminate()
            self._audio_player = None
        self._audio_play_btn.configure(text="▶  Play")

    def _monitor_playback(self, proc: subprocess.Popen):
        """Background thread — resets the button when ffplay finishes naturally."""
        proc.wait()
        self.after(0, self._on_playback_ended, proc)

    def _on_playback_ended(self, proc: subprocess.Popen):
        if self._audio_player is proc:   # only reset if still the active player
            self._audio_player = None
            self._audio_play_btn.configure(text="▶  Play")

    @staticmethod
    def _get_audio_duration(path: Path) -> str:
        """Return duration as 'm:ss' using ffprobe, or empty string on failure."""
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
        choice = self._audio_var.get()
        return self._audio_tracks.get(choice)  # None for "None" or unknown key

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
        self._stop_audio()
        if self._aligner:
            self._aligner.close()
        self.destroy()
