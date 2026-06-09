"""Entry point — launch the Face Sequence Aligner GUI."""

import os
import sys

# MediaPipe's C++ telemetry (clearcut uploader) writes directly to fd 2,
# bypassing Python logging entirely. For a windowed GUI app this is safe
# to suppress — Python exceptions surface as dialogs, not terminal output.
os.environ["GLOG_minloglevel"] = "3"
os.environ["GLOG_logtostderr"] = "0"
try:
    _nul = open(os.devnull, "w")
    os.dup2(_nul.fileno(), sys.stderr.fileno())
    sys.stderr = _nul
except Exception:
    pass

from ui.app import App


def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
