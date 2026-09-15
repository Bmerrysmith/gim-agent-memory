"""Run this checkout without downloading dependencies or installing a package."""

from pathlib import Path
import sys


if __name__ == "__main__":
    # Only this checkout's src directory is added; installed users can run `gim`.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from gim.cli import main

    raise SystemExit(main())
