"""
Main application window.

Layout (top → bottom):
  Menu      — File / Help (custom dark popup menus)
  Toolbar   — folder picker, settings, action buttons, audio
  Preview   — aligned-frame thumbnail strip
  Status    — progress bar + status label

The App class assembles four mixin classes, each in its own module:
  ui.project   — save / load / new project
  ui.alignment — folder picking, face detection, alignment, thumbnail events
  ui.export    — MP4 export pipeline and output panel
  ui.audio     — audio track selection and ffplay preview
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import customtkinter as ctk

try:
    from version import __version__ as _APP_VERSION
except ImportError:
    _APP_VERSION = "unknown"

from aligner import AlignedFrame, FaceAligner
from ui.alignment import AlignMixin
from ui.audio import AudioMixin
from ui.export import ExportMixin
from ui.menu import _PopupMenu
from ui.preview import PreviewStrip
from ui.project import ProjectMixin
from ui.utils import _NO_AUDIO, _RESOLUTIONS, _SORT_OPTIONS, _scan_music


class App(AudioMixin, ExportMixin, AlignMixin, ProjectMixin, ctk.CTk):

    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Face Sequence Aligner")
        self.geometry("1100x680")
        self.minsize(800, 500)

        # Core state
        self._folder: Optional[Path] = None
        self._aligned_frames: list[AlignedFrame] = []
        self._aligner: Optional[FaceAligner] = None
        self._export_thread: Optional[threading.Thread] = None

        # Detection cache — reused when only Face % changes
        self._pending_faces: list[tuple[Path, object]] = []
        self._detected_folder: Optional[Path] = None
        self._aligned_output_size: Optional[tuple[int, int]] = None

        # Audio
        self._audio_tracks: dict[str, Path] = {}
        self._audio_player = None  # subprocess.Popen or None

        # Project / export paths
        self._project_path: Optional[Path] = None
        self._last_export_path: Optional[Path] = None

        # Cancellation token for in-progress raw-preview loads
        self._raw_preview_token: object = object()

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
        file_menu.add_command("New",               self._new_project,    "Ctrl+N")
        file_menu.add_separator()
        file_menu.add_command("Open Project…",     self._open_project,   "Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command("Save Project",      self._save_project,   "Ctrl+S")
        file_menu.add_command("Save Project As…",  self._save_project_as)
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
        dlg.attributes("-toolwindow", True)

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
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(self, corner_radius=0)
        toolbar.grid(row=1, column=0, sticky="ew")
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
        status_bar.grid(row=3, column=0, sticky="ew")
        status_bar.grid_propagate(False)
        self._build_status_bar(status_bar)

    def _build_toolbar(self, parent: ctk.CTkFrame):
        parent.grid_columnconfigure(1, weight=1)

        # Row 0: folder picker + sort
        ctk.CTkButton(
            parent, text="Open Folder…", command=self._pick_folder, width=130
        ).grid(row=0, column=0, padx=(10, 6), pady=(8, 2), sticky="w")

        self._folder_label = ctk.CTkLabel(parent, text="No folder selected", anchor="w")
        self._folder_label.grid(row=0, column=1, padx=(0, 10), pady=(8, 2), sticky="ew")

        ctk.CTkLabel(parent, text="Sort:").grid(row=0, column=2, padx=(0, 4), pady=(8, 2))
        self._sort_var = ctk.StringVar(value="Date ↑")
        ctk.CTkOptionMenu(
            parent, variable=self._sort_var, values=_SORT_OPTIONS, width=130,
            command=lambda _: self._on_sort_changed(),
        ).grid(row=0, column=3, padx=(0, 10), pady=(8, 2))

        # Row 1: settings + action buttons
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
        self._hold_var.trace_add("write", self._update_video_info)
        ctk.CTkEntry(r1, textvariable=self._hold_var, width=52).grid(row=0, column=col, padx=(0, 16))
        col += 1

        ctk.CTkLabel(r1, text="Transition (s):").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._trans_var = ctk.StringVar(value="1.0")
        self._trans_var.trace_add("write", self._update_video_info)
        ctk.CTkEntry(r1, textvariable=self._trans_var, width=52).grid(row=0, column=col, padx=(0, 16))
        col += 1

        ctk.CTkLabel(r1, text="FPS:").grid(row=0, column=col, padx=(0, 2))
        col += 1
        self._fps_var = ctk.StringVar(value="30")
        self._fps_var.trace_add("write", self._update_video_info)
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

        # Row 2: audio
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

        self._audio_duration_label = ctk.CTkLabel(
            r2, text="", width=44, anchor="w", text_color="gray"
        )
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

        self._video_info_label = ctk.CTkLabel(
            parent, text="", anchor="e", text_color="gray", width=140,
        )
        self._video_info_label.grid(row=0, column=1, padx=(0, 8), pady=4)

        self._show_folder_btn = ctk.CTkButton(
            parent, text="Show in Folder", command=self._show_in_folder,
            width=130, state="disabled",
            fg_color="transparent", border_width=1,
            text_color=("#1f538d", "#4da6ff"),
            hover_color=("#e0e8f5", "#1e2d40"),
        )
        self._show_folder_btn.grid(row=0, column=2, padx=(0, 8), pady=4)

        self._progress = ctk.CTkProgressBar(parent, width=200)
        self._progress.set(0)
        self._progress.grid(row=0, column=3, padx=(0, 12), pady=4)

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

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
