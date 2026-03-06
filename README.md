# 📷 PROJECT ARGUS — Claims Scanner

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-yellow)
![License](https://img.shields.io/badge/license-MIT-green)
![Build](https://img.shields.io/github/actions/workflow/status/KINGdeusX/PROJECT_ARGUS/release.yml?label=build)

A desktop OCR scanning tool built with **PyQt6**, **OpenCV**, and **Tesseract**.  
Scan documents via webcam, extract text, and export to CSV — all in one click.

---

## Features

- 🎥 Live webcam preview with adjustable camera settings
- 🔍 OCR extraction via Tesseract (region-selectable)
- 📋 Inline scan history with edit & delete
- 📂 CSV export (append or overwrite)
- 🔄 **Auto-update** — minor updates install in-app; major updates use the standalone updater

---

## Requirements

- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\tesseract.exe`

Install Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Running from Source

```bash
python main.py
```

---

## Building the EXE

Requires [PyInstaller](https://pyinstaller.org/):

```bash
pip install pyinstaller
build.bat
```

Output files in `dist/`:
| File | Description |
|---|---|
| `claims_scanner.exe` | Main application |
| `argus_updater.exe` | Standalone updater (for major releases) |

---

## Update System

| Update type | How it works |
|---|---|
| **Minor / Patch** (`1.0.x`, `1.x.0`) | The app notifies you in the status bar and can install itself automatically |
| **Major** (`2.0.0`, `3.0.0`) | The app prompts you to run `argus_updater.exe`, which downloads and replaces the installation |

Updates are fetched from [GitHub Releases](https://github.com/KINGdeusX/PROJECT_ARGUS/releases).

---

## Releasing a New Version

1. Update `APP_VERSION` in `version.py` and `version.json`
2. Commit changes: `git commit -am "Release vX.Y.Z"`
3. Tag the commit: `git tag vX.Y.Z && git push origin vX.Y.Z`
4. GitHub Actions builds both EXEs and publishes the release automatically

---

## Project Structure

```
PROJECT ARGUS/
├── main.py                 # Main application
├── updater.py              # Standalone major-update installer
├── version.py              # Version constants & helpers
├── version.json            # Runtime version file (read by updater)
├── requirements.txt        # Python dependencies
├── claims_scanner.spec     # PyInstaller spec — main app
├── argus_updater.spec      # PyInstaller spec — updater
├── build.bat               # Build both EXEs in one command
├── .github/
│   └── workflows/
│       └── release.yml     # CI/CD — auto-build on tag push
└── tesseract/              # Bundled Tesseract (optional)
```
