"""Tracking per kamera: ByteTrack via Ultralytics + line-crossing flow (PRD §11.2/11.3).

Satu kendaraan dalam 20 frame = 1 track, bukan 20 hitungan. `flow_count` (kumulatif
garis) TIDAK PERNAH dipakai sebagai ukuran antrean saat ini.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from . import config
from .yolo_engine import YoloEngine


class CameraTracker:
    """State tracking untuk SATU kamera: track ID, posisi terakhir, histori posisi
    untuk deteksi diam, dan penghitung garis per arah."""

    STOP_SPEED_PX_S = 12.0  # di bawah ini = diam (frame 640px, akan diskala)

    def __init__(self, line_y: float = 50.0) -> None:
        self.line_y = line_y  # garis virtual horizontal, 0-100
        self.last_pos: Dict[int, Tuple[float, float, float]] = {}  # tid -> (cx, cy, t)
        self.still_since: Dict[int, float] = {}
        self.flow_total = 0
        self.flow_events: deque = deque()  # timestamp kejadian 60 dtk terakhir
        self._lock = threading.Lock()

    def update(self, tracks: List[Dict[str, Any]], now: float) -> Dict[str, Any]:
        """tracks: [{track_id, cx, cy}] ternormalisasi 0-100."""
        with self._lock:
            seen = set()
            for tr in tracks:
                tid = int(tr["track_id"])
                seen.add(tid)
                cx, cy = float(tr["cx"]), float(tr["cy"])
                prev = self.last_pos.get(tid)
                if prev is not None:
                    _, pcy, pt = prev
                    dt = max(1e-3, now - pt)
                    speed = abs(cy - pcy) / dt  # unit 0-100 per detik
                    crossed = (pcy - self.line_y) * (cy - self.line_y) < 0
                    if crossed:
                        self.flow_total += 1
                        self.flow_events.append(now)
                    if speed < self.STOP_SPEED_PX_S / 6.4:  # dinormalisasi ke skala 0-100
                        self.still_since.setdefault(tid, now)
                    else:
                        self.still_since.pop(tid, None)
                self.last_pos[tid] = (cx, cy, now)
            for tid in [t for t in self.last_pos if t not in seen]:
                self.last_pos.pop(tid, None)
                self.still_since.pop(tid, None)
            cutoff = now - 60.0
            while self.flow_events and self.flow_events[0] < cutoff:
                self.flow_events.popleft()
            still_ids = [t for t, s0 in self.still_since.items() if now - s0 >= 2.0]
            return {
                "tracked": len(self.last_pos),
                "flow_total": self.flow_total,
                "flow_60s": len(self.flow_events),
                "stopped_ids": still_ids,
            }


class TrackerHub:
    """Banyak kamera: satu CameraTracker per camera_id (isolasi MI-04)."""

    def __init__(self, engine: YoloEngine) -> None:
        self.engine = engine
        self._trackers: Dict[str, CameraTracker] = {}
        self._lock = threading.Lock()

    def for_camera(self, camera_id: str) -> CameraTracker:
        with self._lock:
            tr = self._trackers.get(camera_id)
            if tr is None:
                tr = CameraTracker()
                self._trackers[camera_id] = tr
            return tr

    def track_frame(
        self, camera_id: str, frame: np.ndarray,
        roi: Tuple[float, float, float, float] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """YOLO track(persist) + update garis. Return (tracks, info)."""
        assert self.engine.model is not None, "model not loaded"
        t0 = time.monotonic()
        frame = self.engine.upscale_small(frame)
        fh, fw = frame.shape[:2]
        crop, (ox, oy) = self.engine.crop_roi(frame, roi)
        res = self.engine.model.track(
            crop, imgsz=config.INFER_IMGSZ, conf=config.CONFIDENCE, iou=config.IOU,
            persist=True, verbose=False, tracker="bytetrack.yaml",
        )
        tracks: List[Dict[str, Any]] = []
        r = res[0] if res else None
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            ids = r.boxes.id
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            tids = ids.cpu().numpy().astype(int) if ids is not None else [-1] * len(xyxy)
            for (x1, y1, x2, y2), c, cf, tid in zip(xyxy, cls, conf, tids):
                cx = ((x1 + x2) / 2 + ox) / fw * 100.0
                cy = ((y1 + y2) / 2 + oy) / fh * 100.0
                tracks.append({
                    "track_id": int(tid),
                    "class_id": int(c),
                    "label": self.engine.class_names.get(int(c), str(int(c))),
                    "category": self.engine.category_of(int(c)),
                    "confidence": round(float(cf), 3),
                    "cx": round(cx, 1),
                    "cy": round(cy, 1),
                })
        summary = self.for_camera(camera_id).update(tracks, time.monotonic())
        summary["latency_s"] = round(time.monotonic() - t0, 3)
        return tracks, summary
