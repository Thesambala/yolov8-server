"""Tracking per kamera dengan isolasi penuh (PRD F6/F7/F8).

F6: SATU model YOLO shared = SATU modulator Ultralytics predictor = track ID
    bercampur antar kamera. Karena itu hub ini memegang SATU engine (YOLO instance)
    PER camera_id (weights sama, predictor/tracker terisolasi). RAM ekstra ~22 MB
    per kamera aktif — jauh lebih murah daripada salah hitung.
F7: garis hitung ber-hysteresis band +-3.0: jitter di sekitar garis tidak dihitung;
    hanya transisi penuh ATAS->BAWAH / BAWAH->ATAS yang +1 (direction dicatat).
F8: waiting time aktual dari still_since per track (bukan queue*konstanta).
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from . import config

LINE_BAND = 3.0  # setengah-lebar hysteresis band (skala 0-100)
STILL_SPEED = 12.0 / 6.4  # satuan 0-100 per detik
STILL_MIN_S = 2.0


class CameraTracker:
    """State tracking untuk SATU kamera."""

    def __init__(self, line_y: float = 50.0) -> None:
        self.line_y = line_y
        self.last_pos: Dict[int, Tuple[float, float, float]] = {}
        self.still_since: Dict[int, float] = {}
        self.cat: Dict[int, str] = {}  # kategori per track (vehicle/person/unmotorized)
        self.side: Dict[int, int] = {}  # -1 atas, +1 bawah (di luar band)
        self.flow_total = 0
        self.flow_events: deque = deque()
        self.flow_by_direction: Dict[str, int] = {"down": 0, "up": 0}
        self._lock = threading.Lock()

    def _side_of(self, cy: float, prev_side: int) -> int:
        if cy < self.line_y - LINE_BAND:
            return -1
        if cy > self.line_y + LINE_BAND:
            return 1
        return prev_side

    def update(self, tracks: List[Dict[str, Any]], now: float) -> Dict[str, Any]:
        with self._lock:
            seen = set()
            for tr in tracks:
                tid = int(tr["track_id"])
                seen.add(tid)
                cx, cy = float(tr["cx"]), float(tr["cy"])
                category = str(tr.get("category", "vehicle"))
                self.cat[tid] = category
                is_vehicle = category == "vehicle"
                prev = self.last_pos.get(tid)
                old_side = self.side.get(tid, 0)
                new_side = self._side_of(cy, old_side)
                if prev is not None:
                    _, pcy, pt = prev
                    dt = max(1e-3, now - pt)
                    speed = abs(cy - pcy) / dt  # unit 0-100 per detik
                    # Transisi penuh antar sisi = 1 crossing (dengan arah).
                    if old_side != 0 and new_side != 0 and new_side != old_side:
                        self.flow_total += 1
                        self.flow_events.append(now)
                        direction = "down" if new_side > old_side else "up"
                        self.flow_by_direction[direction] += 1
                    # L: hanya vehicle yang boleh masuk still/waiting.
                    if is_vehicle:
                        if speed < STILL_SPEED:
                            self.still_since.setdefault(tid, now)
                        else:
                            self.still_since.pop(tid, None)
                    else:
                        self.still_since.pop(tid, None)
                if new_side != 0:
                    self.side[tid] = new_side
                self.last_pos[tid] = (cx, cy, now)
            for tid in [t for t in self.last_pos if t not in seen]:
                self.last_pos.pop(tid, None)
                self.still_since.pop(tid, None)
                self.side.pop(tid, None)
                self.cat.pop(tid, None)
            cutoff = now - 60.0
            while self.flow_events and self.flow_events[0] < cutoff:
                self.flow_events.popleft()
            # K: waiting HANYA dari stopped vehicle (>= threshold), bukan semua still.
            vehicle_stopped = [
                t for t, s0 in self.still_since.items()
                if now - s0 >= STILL_MIN_S and self.cat.get(t) == "vehicle"
            ]
            vehicle_wait = max(
                [now - self.still_since[t] for t in vehicle_stopped],
                default=0.0,
            )
            return {
                "tracked": len(self.last_pos),
                "flow_total": self.flow_total,
                "flow_60s": len(self.flow_events),
                "flow_by_direction": dict(self.flow_by_direction),
                "stopped_ids": vehicle_stopped,
                "vehicle_stopped_ids": vehicle_stopped,
                "max_waiting_s": round(vehicle_wait, 1),
                "vehicle_max_waiting_s": round(vehicle_wait, 1),
            }


class TrackerHub:
    """Satu ENGINE (YOLO instance) per camera_id -> tracker internal terisolasi."""

    def __init__(self, engine_factory: Callable[[], Any] | None = None) -> None:
        if engine_factory is None:
            from .yolo_engine import YoloEngine

            def _default_factory() -> Any:
                eng = YoloEngine()
                eng.load()
                return eng

            engine_factory = _default_factory
        self._factory = engine_factory
        self._engines: Dict[str, Any] = {}
        self._trackers: Dict[str, CameraTracker] = {}
        self._lock = threading.Lock()

    def engine_for(self, camera_id: str) -> Any:
        with self._lock:
            eng = self._engines.get(camera_id)
            if eng is None:
                eng = self._factory()
                self._engines[camera_id] = eng
            return eng

    def for_camera(self, camera_id: str) -> CameraTracker:
        with self._lock:
            tr = self._trackers.get(camera_id)
            if tr is None:
                tr = CameraTracker()
                self._trackers[camera_id] = tr
            return tr

    def camera_ids(self):
        with self._lock:
            return list(set(self._engines) | set(self._trackers))

    def track_frame(
        self, camera_id: str, frame: np.ndarray,
        roi: Tuple[float, float, float, float] | None = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        engine = self.engine_for(camera_id)
        t0 = time.monotonic()
        frame = engine.upscale_small(frame)
        fh, fw = frame.shape[:2]
        crop, (ox, oy) = engine.crop_roi(frame, roi)
        res = engine.model.track(
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
            names = getattr(engine, "class_names", {})
            for (x1, y1, x2, y2), c, cf, tid in zip(xyxy, cls, conf, tids):
                cx = ((x1 + x2) / 2 + ox) / fw * 100.0
                cy = ((y1 + y2) / 2 + oy) / fh * 100.0
                label = names.get(int(c), str(int(c))) if isinstance(names, dict) else str(int(c))
                cat = engine.category_of(int(c)) if hasattr(engine, "category_of") else "vehicle"
                tracks.append({
                    "track_id": int(tid),
                    "class_id": int(c),
                    "label": label,
                    "category": cat,
                    "confidence": round(float(cf), 3),
                    "cx": round(cx, 1),
                    "cy": round(cy, 1),
                })
        summary = self.for_camera(camera_id).update(tracks, time.monotonic())
        summary["latency_s"] = round(time.monotonic() - t0, 3)
        return tracks, summary
