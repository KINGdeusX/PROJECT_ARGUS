"""
version.py — Single source of truth for PROJECT ARGUS versioning.
Imported by main.py AND updater.py so both always agree on the version.

Versioning scheme:  MAJOR.MINOR.PATCH
  MAJOR — Breaking change; clients MUST use the standalone updater.exe
  MINOR — New features; the running app can update itself in-place.
  PATCH — Bug-fix; the running app can update itself in-place.
"""

APP_VERSION = "2.0.2"
APP_NAME    = "Claims Scanner"
APP_AUTHOR  = "Project Argus"

# GitHub repository (owner/repo) — used to build the Releases API URL.
GITHUB_REPO = "KINGdeusX/PROJECT_ARGUS"

# Name of the standalone major-updater executable (must match argus_updater.spec).
UPDATER_EXE = "argus_updater.exe"

# GitHub Releases API endpoint (auto-built from GITHUB_REPO)
RELEASES_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"


# ── Helpers ──────────────────────────────────────────────────────────────────

def parse_version(version_str: str) -> tuple[int, int, int]:
    """Parse "MAJOR.MINOR.PATCH" (with optional leading 'v') into a 3-tuple."""
    v = version_str.lstrip("v").strip()
    parts = v.split(".")
    try:
        major = int(parts[0]) if len(parts) > 0 else 0
        minor = int(parts[1]) if len(parts) > 1 else 0
        patch = int(parts[2]) if len(parts) > 2 else 0
    except ValueError:
        major, minor, patch = 0, 0, 0
    return major, minor, patch


def is_newer(remote_ver: str, local_ver: str | None = None) -> bool:
    """Return True if *remote_ver* is strictly newer than *local_ver* (or APP_VERSION)."""
    if local_ver is None:
        local_ver = APP_VERSION
    return parse_version(remote_ver) > parse_version(local_ver)


def is_major_bump(remote_ver: str, local_ver: str | None = None) -> bool:
    """Return True if the remote release has a higher MAJOR than the local version."""
    if local_ver is None:
        local_ver = APP_VERSION
    remote_major, _, _ = parse_version(remote_ver)
    local_major, _, _ = parse_version(local_ver)
    return remote_major > local_major


def version_tuple() -> tuple[int, int, int]:
    """Return the current app version as a (major, minor, patch) tuple."""
    return parse_version(APP_VERSION)
