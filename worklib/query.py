"""Legacy entrypoint kept for backward compatibility.

The canonical query CLI lives in ``worklib/query/__init__.py``.
"""

from __future__ import annotations

from worklib.query import main


if __name__ == "__main__":
    main()
