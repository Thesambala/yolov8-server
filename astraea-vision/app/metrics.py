"""Metrik per kamera/pendekatan + freshness dari FRAME (diterima), bukan inferensi.

Source of truth online/fresh (F): umur frame terakhir DITERIMA (STORE.age_s).
last_infer_at hanya diagnostik. State: intersection -> approach -> camera (MI-04).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import config
from .frame_store import STORE
from .schemas import empty_camera_metrics


def intersection_vision_state(fresh_flags: List[bool]) -> Tuple[str, int]:
    """NORMAL semua fresh, DEGRADED sebagian, FALLBACK nihil."""
    n = sum(1 for f in fresh_flags if f)
    if n == 0:
        return "FALLBACK", 0
    if n == len(fresh_flags) and len(fresh_flags) > 0:
        return "NORMAL", n
    return "DEGRADED", n


class MetricsHub:
    """State: intersection -> approach -> camera. (MI-04: tidak ada global tunggal.)"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # meta[camera_id] = {intersection_id, approach_id, last_infer_at, last_counts...}
        self._meta: Dict[str, Dict[str, Any]] = {}
        # sensor[(intersection_id, approach_id)] = {sensor_level, ir, us, at}
        self._sensor: Dict[tuple, Dict[str, Any]] = {}
        # socket ingest per kamera (True = WSS session aktif)
        self._connected: Dict[str, bool] = {}

    def register(self, camera_id: str, intersection_id: str, approach_id: str) -> None:
        with self._lock:
            m = self._meta.setdefault(camera_id, {})
            m.update({"intersection_id": intersection_id, "approach_id": approach_id})

    def update_inference(
        self,
        camera_id: str,
        active_vehicles: int,
        queue_vehicles: int,
        stopped_vehicles: int,
        flow_60s: int,
        waiting_s: float,
        confidence: float,
    ) -> None:
        with self._lock:
            m = self._meta.setdefault(camera_id, {})
            m.update({
                "last_infer_at": time.monotonic(),
                "active_vehicle_count": active_vehicles,
                "queue_vehicle_count": queue_vehicles,
                "stopped_vehicle_count": stopped_vehicles,
                "flow_count_60s": flow_60s,
                "max_waiting_time_s": round(waiting_s, 1),
                "confidence": round(confidence, 3),
                "online": True,
            })

    def update_sensor(
        self, intersection_id: str, approach_id: str,
        sensor_level: int, ir_occupied: bool, ultrasonic_occupied: bool,
    ) -> None:
        with self._lock:
            self._sensor[(intersection_id, approach_id)] = {
                "sensor_level": int(sensor_level),
                "ir_occupied": bool(ir_occupied),
                "ultrasonic_occupied": bool(ultrasonic_occupied),
                "at": time.monotonic(),
            }

    def set_connected(self, camera_id: str, connected: bool) -> None:
        with self._lock:
            self._connected[camera_id] = bool(connected)

    def camera_state(self, camera_id: str) -> Dict[str, Any]:
        with self._lock:
            m = dict(self._meta.get(camera_id, {}))
            connected = self._connected.get(camera_id)
        # FRESHNESS DARI FRAME (diterima), bukan dari inferensi (H).
        frame_age = STORE.age_s(camera_id)
        frame_fresh = frame_age is not None and frame_age <= config.VISION_FRESH_S
        out = empty_camera_metrics()
        out.update({
            "active_vehicle_count": m.get("active_vehicle_count", 0),
            "flow_count_60s": m.get("flow_count_60s", 0),
            "queue_vehicle_count": m.get("queue_vehicle_count", 0),
            "stopped_vehicle_count": m.get("stopped_vehicle_count", 0),
            "max_waiting_time_s": m.get("max_waiting_time_s", 0.0),
            "confidence": m.get("confidence", 0.0),
            "online": frame_fresh,
            "fresh": frame_fresh,
            "connected": connected,
            "frame_age_s": round(frame_age, 1) if frame_age is not None else None,
            "inference_fresh": bool(m.get("last_infer_at")) and (
                time.monotonic() - m["last_infer_at"] <= config.VISION_FRESH_S
            ),
        })
        return out

    def approach_state(self, intersection_id: str, approach_id: str) -> Dict[str, Any]:
        """Gabungan kamera + sensor sesuai kontrak PRD §15. Salah satu boleh absen."""
        with self._lock:
            cam_id = next(
                (c for c, m in self._meta.items()
                 if m.get("intersection_id") == intersection_id and m.get("approach_id") == approach_id),
                None,
            )
            sens = dict(self._sensor.get((intersection_id, approach_id), {}))
        camera = self.camera_state(cam_id) if cam_id else dict(empty_camera_metrics(), fresh=False)
        sensor = {
            "ir_occupied": sens.get("ir_occupied", False),
            "ultrasonic_occupied": sens.get("ultrasonic_occupied", False),
            "sensor_level": sens.get("sensor_level", 0),
            "healthy": bool(sens),
        }
        diag = None
        if camera.get("online") and sens:
            if camera["active_vehicle_count"] == 0 and sensor["sensor_level"] >= 2:
                diag = "camera-empty-but-sensor-L2"
            elif camera["queue_vehicle_count"] > 0 and sensor["sensor_level"] == 0:
                diag = "camera-queue-but-sensor-clear"
        return {
            "approach_id": approach_id,
            "camera": camera,
            "sensor": sensor,
            "fusion_disagreement": diag,  # diagnostik saja, tidak mengendalikan lampu (§48)
        }

    def approaches_of(self, intersection_id: str):
        with self._lock:
            out = {m["approach_id"] for m in self._meta.values()
                   if m.get("intersection_id") == intersection_id and m.get("approach_id")}
            out |= {a for (i, a) in self._sensor if i == intersection_id}
        return sorted(out)


HUB = MetricsHub()
