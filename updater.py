"""
updater.py  --  PROJECT ARGUS Standalone Major Updater
=======================================================
This is compiled separately as argus_updater.exe via argus_updater.spec.
It is launched by the main app when a MAJOR version bump is detected
(e.g. v1.x.x → v2.0.0).

Flow:
  1. Fetch latest release info from GitHub Releases API
  2. Show the changelog / release notes
  3. Download the new claims_scanner.exe with a progress bar
  4. Replace the old EXE (handles the Windows file-lock by renaming)
  5. Write the new version.json
  6. Optionally relaunch the updated app
"""

import sys
import os
import json
import shutil
import hashlib
import tempfile
import subprocess
import urllib.request
import urllib.error
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QProgressBar, QTextEdit,
    QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPalette

# ── Import shared version info ────────────────────────────────────────────────
try:
    from version import (
        APP_NAME, APP_VERSION, GITHUB_REPO,
        RELEASES_API_URL, parse_version, is_newer
    )
except ImportError:
    # Fallback if run standalone without version.py nearby
    APP_NAME        = "Claims Scanner"
    APP_VERSION     = "1.0.0"
    GITHUB_REPO     = "KINGdeusX/PROJECT_ARGUS"
    RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

    def parse_version(v):
        v = v.lstrip("v").strip()
        parts = v.split(".")
        try:
            return tuple(int(x) for x in (parts + ["0", "0", "0"])[:3])
        except ValueError:
            return (0, 0, 0)

    def is_newer(remote, local=APP_VERSION):
        return parse_version(remote) > parse_version(local)


MAIN_EXE_NAME = "claims_scanner.exe"
VERSION_FILE  = "version.json"


# ── Worker thread: fetch release info ─────────────────────────────────────────
class FetchReleaseThread(QThread):
    done    = pyqtSignal(dict)    # emits release info dict on success
    error   = pyqtSignal(str)     # emits error message on failure

    def run(self):
        try:
            req = urllib.request.Request(
                RELEASES_API_URL,
                headers={"User-Agent": f"ProjectArgus-Updater/{APP_VERSION}",
                         "Accept": "application/vnd.github+json"}
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            self.done.emit(data)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Worker thread: download asset ─────────────────────────────────────────────
class DownloadThread(QThread):
    progress = pyqtSignal(int)   # 0-100
    done     = pyqtSignal(str)   # path to downloaded temp file
    error    = pyqtSignal(str)

    def __init__(self, url: str, parent=None):
        super().__init__(parent)
        self.url = url

    def run(self):
        try:
            req = urllib.request.Request(
                self.url,
                headers={"User-Agent": f"ProjectArgus-Updater/{APP_VERSION}"}
            )
            tmp = tempfile.mktemp(suffix=".exe")
            with urllib.request.urlopen(req, timeout=60) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536
                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = int(downloaded * 100 / total)
                            self.progress.emit(min(pct, 99))
            self.progress.emit(100)
            self.done.emit(tmp)
        except Exception as exc:
            self.error.emit(str(exc))


# ── Main updater window ────────────────────────────────────────────────────────
class UpdaterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} — Updater")
        self.setMinimumSize(540, 420)
        self.setStyleSheet(STYLESHEET)

        self._release_info = None
        self._download_url = None
        self._remote_version = None

        self._build_ui()
        # Auto-start check after a short delay
        QTimer.singleShot(400, self._check_for_update)

    # ── UI ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        title = QLabel(f"🔄  {APP_NAME}  —  Updater")
        title.setObjectName("title")
        layout.addWidget(title)

        self.status_label = QLabel("Checking for updates…")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        version_row = QHBoxLayout()
        self.current_ver_lbl = QLabel(f"Installed:  v{APP_VERSION}")
        self.current_ver_lbl.setObjectName("verLabel")
        self.remote_ver_lbl  = QLabel("Latest:  —")
        self.remote_ver_lbl.setObjectName("verLabel")
        version_row.addWidget(self.current_ver_lbl)
        version_row.addStretch()
        version_row.addWidget(self.remote_ver_lbl)
        layout.addLayout(version_row)

        self.notes_area = QTextEdit()
        self.notes_area.setReadOnly(True)
        self.notes_area.setPlaceholderText("Release notes will appear here…")
        self.notes_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.notes_area)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.install_btn = QPushButton("⬇  Download & Install")
        self.install_btn.setObjectName("primaryBtn")
        self.install_btn.setEnabled(False)
        self.install_btn.clicked.connect(self._start_download)

        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.close)

        btn_row.addWidget(self.install_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

    # ── Update check ────────────────────────────────────────────────────────
    def _check_for_update(self):
        self.status_label.setText("Fetching latest release info from GitHub…")
        self._fetch_thread = FetchReleaseThread(self)
        self._fetch_thread.done.connect(self._on_release_fetched)
        self._fetch_thread.error.connect(self._on_fetch_error)
        self._fetch_thread.start()

    def _on_release_fetched(self, data: dict):
        tag = data.get("tag_name", "")
        self._remote_version = tag.lstrip("v")
        self._release_info   = data

        self.remote_ver_lbl.setText(f"Latest:  {tag}")
        body = data.get("body", "_No release notes provided._")
        self.notes_area.setMarkdown(body)

        if not is_newer(self._remote_version):
            self.status_label.setText(
                "✅  You are already running the latest version."
            )
            return

        # Find the main EXE asset
        for asset in data.get("assets", []):
            if asset.get("name", "").lower() == MAIN_EXE_NAME.lower():
                self._download_url = asset["browser_download_url"]
                break

        if not self._download_url:
            self.status_label.setText(
                f"⚠  Update {tag} found but no '{MAIN_EXE_NAME}' asset attached."
            )
            return

        self.status_label.setText(
            f"🆕  Version {tag} is available!  "
            f"Click 'Download & Install' to update."
        )
        self.install_btn.setEnabled(True)

    def _on_fetch_error(self, msg: str):
        self.status_label.setText(f"❌  Could not fetch release info: {msg}")

    # ── Download ─────────────────────────────────────────────────────────────
    def _start_download(self):
        if not self._download_url:
            return
        self.install_btn.setEnabled(False)
        self.close_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Downloading update…")

        self._dl_thread = DownloadThread(self._download_url, self)
        self._dl_thread.progress.connect(self.progress_bar.setValue)
        self._dl_thread.done.connect(self._on_download_done)
        self._dl_thread.error.connect(self._on_download_error)
        self._dl_thread.start()

    def _on_download_done(self, tmp_path: str):
        self.status_label.setText("Installing…")
        try:
            self._install(tmp_path)
        except Exception as exc:
            self._on_download_error(str(exc))
            return

        # Write new version.json
        self._write_version_json()

        self.status_label.setText(
            f"✅  Updated to v{self._remote_version}!  "
            "Restart the app to use the new version."
        )
        self.progress_bar.setValue(100)

        reply = QMessageBox.question(
            self, "Update Complete",
            f"Claims Scanner has been updated to v{self._remote_version}.\n\n"
            "Launch the updated app now?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._launch_app()
        self.close_btn.setEnabled(True)

    def _on_download_error(self, msg: str):
        self.status_label.setText(f"❌  Download failed: {msg}")
        self.install_btn.setEnabled(True)
        self.close_btn.setEnabled(True)
        self.progress_bar.setVisible(False)

    # ── Install helpers ──────────────────────────────────────────────────────
    def _install(self, tmp_path: str):
        """Replace the running EXE (or main EXE in same dir) with the downloaded one."""
        # Determine where to place the new EXE
        if getattr(sys, "frozen", False):
            # We are the updater EXE; the main app is in the same directory
            app_dir = Path(sys.executable).parent
        else:
            # Running as a script (dev mode) — place next to main.py
            app_dir = Path(__file__).parent

        target = app_dir / MAIN_EXE_NAME
        backup = app_dir / (MAIN_EXE_NAME + ".bak")

        # Back up old version
        if target.exists():
            shutil.copy2(str(target), str(backup))

        try:
            shutil.move(tmp_path, str(target))
        except PermissionError:
            # If the old EXE is locked (running), rename it first
            if target.exists():
                locked = app_dir / (MAIN_EXE_NAME + ".old")
                target.rename(locked)
            shutil.move(tmp_path, str(target))

    def _write_version_json(self):
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).parent

        version_file = app_dir / VERSION_FILE
        data = {
            "version": self._remote_version,
            "app_name": APP_NAME,
            "github_repo": GITHUB_REPO
        }
        try:
            version_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            pass  # non-fatal

    def _launch_app(self):
        if getattr(sys, "frozen", False):
            app_dir = Path(sys.executable).parent
        else:
            app_dir = Path(__file__).parent
        exe = app_dir / MAIN_EXE_NAME
        if exe.exists():
            subprocess.Popen([str(exe)], cwd=str(app_dir))
        self.close()


# ── Stylesheet ─────────────────────────────────────────────────────────────────
STYLESHEET = """
QWidget {
    background-color: #1a1d23;
    color: #e0e4ef;
    font-family: "Segoe UI", Inter, sans-serif;
    font-size: 13px;
}
QLabel#title {
    font-size: 18px;
    font-weight: 700;
    color: #7ecfff;
    padding-bottom: 4px;
}
QLabel#status {
    color: #c0c8e0;
    font-size: 13px;
}
QLabel#verLabel {
    color: #a0b0cc;
    font-size: 12px;
}
QTextEdit {
    background-color: #12151c;
    border: 1px solid #2e3448;
    border-radius: 6px;
    padding: 8px;
    color: #d0d8f0;
}
QProgressBar {
    border: 1px solid #2e3448;
    border-radius: 5px;
    background: #12151c;
    height: 16px;
    text-align: center;
    color: #e0e4ef;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f8ef7, stop:1 #7ecfff);
    border-radius: 4px;
}
QPushButton {
    background-color: #262b38;
    border: 1px solid #3a4055;
    border-radius: 6px;
    padding: 7px 18px;
    color: #c8d4f0;
}
QPushButton:hover  { background-color: #2e3448; }
QPushButton:pressed{ background-color: #3d4560; }
QPushButton#primaryBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #4f8ef7, stop:1 #6c63ff);
    color: #ffffff;
    font-weight: 600;
    border: none;
}
QPushButton#primaryBtn:hover  { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #6aa3ff, stop:1 #8077ff); }
QPushButton#primaryBtn:pressed{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #3d76e0, stop:1 #574fff); }
QPushButton:disabled { color: #4a5270; border-color: #252a38; }
"""


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = UpdaterWindow()
    window.show()
    sys.exit(app.exec())
