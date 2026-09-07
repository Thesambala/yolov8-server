"""Orkestrator astraea-vision-service: model -> mqtt -> ingest -> inferensi -> fuzzy.

Frekuensi (§49): frame 2-5 FPS per kamera, metrik ~1 Hz, fuzzy 1/detik, rekomendasi
hanya berubah bermakna (anti-flicker §50: deadband 2 detik pada green).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict

import cv2
import numpy as np
import uvicorn

from . import camera_ingest, config, fuzzy, mqtt
from .controller_registry import controller_for
from .frame_store import STORE
from .metrics import HUB
from .schemas import build_recommendation
from .stream import MODEL_INFO, app as fastapi_app
from .tracking import TrackerHub
from .yolo_engine import YoloEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vision.main")

ENGINE = YoloEngine()
TRACKERS: TrackerHub | None = None
_last_green: Dict[tuple, tuple] = {}  # (iid, approach) -> (value, t)


def on_sensor_payload(intersection_id: str, origin: str, payload: Dict[str, Any]) -> None:
    """Telemetri controller (kanonis/legacy) -> level sensor per pendekatan."""
    try:
        if origin == "legacy":
            for lane in ("north", "south", "east", "west"):
                if f"{lane}_density_level" in payload:
                    HUB.update_sensor(
                        intersection_id, lane,
                        int(payload.get(f"{lane}_density_level", 0)),
                        bool(payload.get(f"{lane}_vehicle_detected", False)),
                        bool(payload.get(f"{lane}_ultrasonic_detected", False)),
                    )
        else:
            for approach_id, st in (payload.get("approaches") or {}).items():
                HUB.update_sensor(
                    intersection_id, str(approach_id),
                    int(st.get("sensor_level", 0)),
                    bool(st.get("ir_occupied", False)),
                    bool(st.get("ultrasonic_occupied", False)),
                )
    except (ValueError, TypeError, AttributeError) as exc:
        logger.warning("bad sensor payload: %s", exc)


def stable_green(iid: str, approach: str, value: float) -> float:
    """Deadband 2 detik: rekomendasi aktif tidak flicker (§50)."""
    key = (iid, approach)
    now = time.monotonic()
    prev = _last_green.get(key)
    if prev is None or abs(value - prev[0]) >= 2.0 or now - prev[1] > 10.0:
        _last_green[key] = (value, now)
        return value
    return prev[0]


def waiting_actual(tracker_summary: Dict[str, Any]) -> float:
    """G: waiting MURNI dari track (max durasi diam). Tanpa queue*konstanta.
    0.0 bila belum ada stopped track yang memenuhi threshold."""
    return round(float(tracker_summary.get("max_waiting_s", 0.0) or 0.0), 1)


def inference_loop() -> None:
    assert TRACKERS is not None
    period = 1.0 / max(0.5, config.METRIC_HZ * 3.0)  # ~3x metrik, latest-frame-only
    while True:
        t0 = time.monotonic()
        for camera_id in STORE.camera_ids():
            item = STORE.get(camera_id)
            if not item:
                continue
            arr = np.frombuffer(item["jpeg"], dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                tracks, summary = TRACKERS.track_frame(camera_id, img)
            except Exception as exc:
                logger.warning("infer %s failed: %s", camera_id, exc)
                continue
            vehicles = [t for t in tracks if t["category"] == "vehicle"]
            stopped = set(summary.get("stopped_ids", []))
            queue_n = sum(1 for t in vehicles if t["track_id"] in stopped)
            confs = [t["confidence"] for t in vehicles]
            HUB.update_inference(
                camera_id,
                active_vehicles=len(vehicles),
                queue_vehicles=queue_n,
                stopped_vehicles=len(stopped),
                flow_60s=summary.get("flow_60s", 0),
                waiting_s=waiting_actual(summary),
                confidence=sum(confs) / len(confs) if confs else 0.0,
            )
        dt = time.monotonic() - t0
        time.sleep(max(0.05, period - dt))


def fuzzy_loop() -> None:
    tick = 0
    while True:
        t0 = time.monotonic()
        tick += 1
        iids = set()
        for cid in STORE.camera_ids():
            m = HUB._meta.get(cid, {})
            if m.get("intersection_id"):
                iids.add(m["intersection_id"])
        for iid in sorted(iids):
            approaches: Dict[str, Any] = {}
            for approach in HUB.approaches_of(iid):
                st = HUB.approach_state(iid, approach)
                cam = st["camera"]
                cap = config.QUEUE_CAPACITY_REF
                qratio = min(1.0, cam.get("queue_vehicle_count", 0) / cap) if cap > 0 else 0.0
                rec = fuzzy.recommend_green(
                    qratio, cam.get("max_waiting_time_s", 0.0),
                    st["sensor"].get("sensor_level", 0), cam.get("online", False),
                    config.MIN_GREEN_S, config.MAX_GREEN_S,
                )
                green = stable_green(iid, approach, rec["recommended_green_s"])
                approaches[approach] = {
                    "online": bool(cam.get("online", False)),
                    "active_vehicle_count": cam.get("active_vehicle_count", 0),
                    "queue_vehicle_count": cam.get("queue_vehicle_count", 0),
                    "sensor_level": st["sensor"].get("sensor_level", 0),
                    "waiting_time_s": cam.get("max_waiting_time_s", 0.0),
                    "recommended_green_s": green,
                }
            if not approaches:
                continue
            rec_msg = build_recommendation(
                iid, controller_for(iid), approaches,
                schema_version=config.SCHEMA_VERSION,
                valid_for_ms=config.RECOMMENDATION_VALID_MS,
            )
            topics = mqtt.canonical_topics(iid)
            mqtt.publish(topics["recommendation"], rec_msg)
            mqtt.publish(topics["vision"], {
                "intersection_id": iid, "approaches": approaches,
                "generated_at": rec_msg["generated_at"],
            })
            # Kompatibilitas legacy 5 detikan agar dashboard lama tetap hidup (§26)
            # tanpa membanjiri DynamoDB/S3 (kanonis tetap 1 Hz, sampled di subscriber).
            if tick % 5 == 0:
                mqtt.publish_legacy_compat(iid, f"CAM_YOLO_{iid}", approaches)
        dt = time.monotonic() - t0
        time.sleep(max(0.2, config.FUZZY_MIN_INTERVAL_S - dt))


def main() -> None:
    global TRACKERS
    info = ENGINE.load()
    MODEL_INFO.update(info)
    # F6: engine pertama dipakai ulang untuk kamera pertama (tetap 1 instance
    # per kamera); kamera berikutnya me-load instance sendiri (terisolasi).
    first = [ENGINE]

    def _factory():
        if first:
            return first.pop()
        eng = YoloEngine()
        eng.load()
        return eng

    TRACKERS = TrackerHub(engine_factory=_factory)
    mqtt.start(on_sensor=on_sensor_payload)
    threading.Thread(target=inference_loop, daemon=True, name="inference").start()
    threading.Thread(target=fuzzy_loop, daemon=True, name="fuzzy").start()
    threading.Thread(
        target=lambda: asyncio.run(camera_ingest.run_forever()),
        daemon=True, name="ingest",
    ).start()
    logger.info("astraea-vision-service up: ws=%d http=%d", config.WS_PORT, config.HTTP_PORT)
    uvicorn.run(fastapi_app, host=config.HTTP_HOST, port=config.HTTP_PORT, log_level="info")


if __name__ == "__main__":
    main()
