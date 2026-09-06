"""Frame store: JPEG terakhir per kamera + meta, thread-safe (PRD §10/46)."""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, Dict[str, Any]] = {}

    def put(
        self,
        camera_id: str,
        jpeg: bytes,
        width: int = 0,
        height: int = 0,
        rssi: int = 0,
    ) -> None:
        with self._lock:
            self._frames[camera_id] = {
                "jpeg": jpeg,
                "at": time.monotonic(),
                "wall": time.time(),
                "bytes": len(jpeg),
                "width": width,
                "height": height,
                "rssi": rssi,
            }

    def get(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            item = self._frames.get(camera_id)
            return dict(item) if item else None

    def age_s(self, camera_id: str) -> Optional[float]:
        with self._lock:
            item = self._frames.get(camera_id)
            if not item:
                return None
            return time.monotonic() - item["at"]

    def camera_ids(self):
        with self._lock:
            return list(self._frames.keys())


STORE = FrameStore()
