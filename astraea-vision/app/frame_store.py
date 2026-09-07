"""Frame store: JPEG terakhir per kamera + seq monotonik + umur frame (E/G/H).

Hanya latest frame (tanpa antrean). `claim_frame` menjamin satu frame fisik
tidak diinferensi dua kali dan frame basi tidak diinferensi.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frames: Dict[str, Dict[str, Any]] = {}
        self._seq: Dict[str, int] = {}
        self._claimed: Dict[str, int] = {}

    def put(
        self,
        camera_id: str,
        jpeg: bytes,
        width: int = 0,
        height: int = 0,
        rssi: int = 0,
    ) -> int:
        """Simpan frame terbaru; return seq monotonikbaru. Thread-safe."""
        with self._lock:
            seq = self._seq.get(camera_id, 0) + 1
            self._seq[camera_id] = seq
            self._frames[camera_id] = {
                "jpeg": jpeg,
                "at": time.monotonic(),
                "wall": time.time(),
                "bytes": len(jpeg),
                "width": width,
                "height": height,
                "rssi": rssi,
                "seq": seq,
            }
            return seq

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

    def claim_frame(self, camera_id: str, max_age_s: float) -> Optional[Dict[str, Any]]:
        """Ambil frame HANYA bila seq lebih baru dari klaim terakhir DAN umur
        frame <= max_age_s. Return None bila tidak ada kerjaan (skip infer)."""
        with self._lock:
            item = self._frames.get(camera_id)
            if not item:
                return None
            if item["seq"] <= self._claimed.get(camera_id, 0):
                return None  # frame sama sudah diproses
            if time.monotonic() - item["at"] > max_age_s:
                return None  # frame basi, jangan infer ulang
            self._claimed[camera_id] = item["seq"]
            return dict(item)

    def camera_ids(self):
        with self._lock:
            return list(self._frames.keys())


STORE = FrameStore()
