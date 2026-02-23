"""Library operations module.

This package refactors and replaces the standalone scripts:
- rebuild_library_versioned.py
- fill_category_vectorstores.py
- fill_category_keywords.py
- delete_old_vectorstore_files.py

It is designed to live alongside your existing `worklib` package and reuse it when possible.
"""

from __future__ import annotations
