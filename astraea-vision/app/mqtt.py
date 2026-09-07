"""MQTT: publish metrik+rekomendasi kanonis & legacy; subscribe telemetri sensor (PRD §25/26/51)."""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import paho.mqtt.client as mqtt

from . import config

logger = logging.getLogger("vision.mqtt")

_client: Optional[mqtt.Client] = None
_lock = threading.Lock()
_sensor_cb: Optional[Callable[[str, str, Dict[str, Any]], None]] = None


def canonical_topics(intersection_id: str) -> Dict[str, str]:
    b = f"astraea/v1/intersections/{intersection_id}"
    return {
        "vision": f"{b}/vision/metrics",
        "recommendation": f"{b}/control/recommendation",
        "telemetry": f"{b}/controllers/+/telemetry",
        "status": f"{b}/controllers/+/status",
    }


def _on_message(client, userdata, msg) -> None:
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return
    if _sensor_cb is None:
        return
    parts = msg.topic.split("/")
    # astraea/v1/intersections/{iid}/controllers/{cid}/telemetry|status
    # traffic/{device}/data  (legacy)
    try:
        if parts[0] == "astraea" and len(parts) >= 7:
            iid = parts[3]
            _sensor_cb(iid, "canonical", payload)
        elif parts[0] == "traffic" and len(parts) >= 3:
            iid = str(payload.get("intersection_id") or payload.get("intersectionId")
                      or "SIMPANG_TALUN_01")
            _sensor_cb(iid, "legacy", payload)
    except Exception as exc:
        logger.warning("sensor-cb failed: %s", exc)


def start(on_sensor=None) -> mqtt.Client:
    global _client, _sensor_cb
    _sensor_cb = on_sensor
    with _lock:
        if _client is not None:
            return _client
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        if config.MQTT_USER:
            c.username_pw_set(config.MQTT_USER, config.MQTT_PASS)
        c.on_message = _on_message
        c.connect(config.MQTT_HOST, config.MQTT_PORT, config.MQTT_KEEPALIVE)
        c.subscribe("astraea/v1/intersections/+/controllers/+/telemetry", qos=0)
        c.subscribe("traffic/+/data", qos=0)  # kompatibilitas legacy §26
        c.loop_start()
        _client = c
        logger.info("MQTT connected %s:%d", config.MQTT_HOST, config.MQTT_PORT)
        return c


def publish(topic: str, payload: Dict[str, Any]) -> bool:
    if _client is None:
        return False
    try:
        _client.publish(topic, json.dumps(payload), qos=0)
        return True
    except Exception as exc:
        logger.warning("publish failed %s: %s", topic, exc)
        return False


def publish_legacy_compat(intersection_id: str, device_id: str, approaches: Dict[str, Any]) -> bool:
    """Jembatan legacy: agregat per-pendekatan -> traffic/{device}/data (§26).

    CATATAN SKEMA (E): main.py mengirim approaches FLAT per pendekatan
    (online, active_vehicle_count, queue_vehicle_count, sensor_level, ...),
    BUKAN nested camera{}/sensor{}. Jangan baca nested (hasilnya selalu 0).
    queue_vehicle_count (KENDARAAN) tidak pernah dipetakan ke cm.
    """
    flat: Dict[str, Any] = {
        "device_id": device_id,
        "intersection_id": intersection_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "astraea-vision",
    }
    for lane, st in approaches.items():
        if not isinstance(st, dict):
            continue
        flat[f"{lane}_vehicle_count"] = st.get("active_vehicle_count", 0)
        flat[f"{lane}_density_level"] = st.get("sensor_level", 0)
        flat[f"{lane}_queue_vehicles"] = st.get("queue_vehicle_count", 0)
        flat[f"{lane}_queue_estimate_cm"] = 0  # tak ada cm dari vision; jangan karang
        flat[f"{lane}_vision_online"] = st.get("online", False)
    return publish(f"traffic/{device_id}/data", flat)
