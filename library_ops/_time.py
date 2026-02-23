from __future__ import annotations
import time

def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")
