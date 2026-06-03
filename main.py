"""Entry point — launch the Face Sequence Aligner GUI."""

import sys

import customtkinter as ctk

from ui.app import App


def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
