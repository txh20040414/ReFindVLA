"""Package entrypoint.

Supports both:
- `python -m main_interactive_track.main` from `Code/`
- `python main.py` from inside `main_interactive_track/`
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_utf8_console():
    """Make Windows console output tolerate emoji/unicode logs."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8")
            except Exception:
                pass


def _load_main():
    """Import the real entrypoint in a way that works for direct script execution."""
    try:
        from .recover_vla import main as entry_main
        return entry_main
    except ImportError:
        package_dir = Path(__file__).resolve().parent
        code_dir = package_dir.parent
        if str(code_dir) not in sys.path:
            sys.path.insert(0, str(code_dir))
        from main_interactive_track.recover_vla import main as entry_main
        return entry_main


main = _load_main()


if __name__ == "__main__":
    _ensure_utf8_console()
    main()
