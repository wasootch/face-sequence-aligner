"""Custom dark-themed popup menu.

tk.Menu ignores bg/fg on Windows 11 (the OS renders it), so we build our own
using a borderless Toplevel with Frame/Label widgets.
"""

from __future__ import annotations

import tkinter as tk
import customtkinter as ctk


class _PopupMenu:
    _W         = 220
    _BG        = "#2b2b2b"
    _BORDER    = "#555555"
    _FG        = "#dce4ee"
    _FG_DIM    = "#666666"
    _HOVER_BG  = "#1f538d"
    _SEP_COLOR = "#444444"

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
                        w.bind("<Enter>",    lambda e, _ws=ws: [_w.configure(bg=hbg) for _w in _ws])
                        w.bind("<Leave>",    lambda e, _ws=ws: [_w.configure(bg=bg)  for _w in _ws])
                        w.bind("<Button-1>", lambda e, _c=cmd: close(_c))

                _bind(widgets, command)

        popup.after(150, lambda: popup.bind("<FocusOut>", lambda e: close()))
        popup.bind("<Escape>", lambda e: close())

        popup.update_idletasks()
        h  = popup.winfo_reqheight()
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        popup.geometry(f"{self._W}x{h}+{min(x, sw - self._W)}+{min(y, sh - h)}")
        popup.deiconify()
        popup.lift()
        popup.focus_force()
