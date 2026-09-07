"""Test MQTT legacy compat: skema flat main.py -> payload legacy benar (E/F).

Publish di-mock agar tanpa broker.
"""
import sys

sys.path.insert(0, ".")

import app.mqtt as mqtt

CAPTURED = []


def fake_publish(topic, payload):
    CAPTURED.append((topic, payload))
    return True


mqtt._client = object()  # agar publish() tidak early-return False
mqtt.publish = fake_publish


def sample_approaches():
    return {
        "north": {"online": True, "active_vehicle_count": 8, "queue_vehicle_count": 5,
                  "sensor_level": 2, "waiting_time_s": 12.4, "recommended_green_s": 40},
        "south": {"online": True, "active_vehicle_count": 3, "queue_vehicle_count": 1,
                  "sensor_level": 1, "waiting_time_s": 4.0, "recommended_green_s": 25},
        "east": {"online": False, "active_vehicle_count": 1, "queue_vehicle_count": 0,
                 "sensor_level": 0, "waiting_time_s": 0.0, "recommended_green_s": 14},
    }


def test_legacy_counts_not_zero():
    CAPTURED.clear()
    assert mqtt.publish_legacy_compat("SIMPANG_TALUN_01", "CAM_X", sample_approaches()) is True
    assert len(CAPTURED) == 1
    topic, flat = CAPTURED[0]
    assert topic == "traffic/CAM_X/data"
    assert flat["north_vehicle_count"] == 8
    assert flat["south_vehicle_count"] == 3
    assert flat["east_vehicle_count"] == 1
    assert flat["north_density_level"] == 2
    assert flat["south_density_level"] == 1
    assert flat["east_density_level"] == 0


def test_queue_not_centimeter():
    CAPTURED.clear()
    mqtt.publish_legacy_compat("I", "D", sample_approaches())
    _, flat = CAPTURED[0]
    assert flat["north_queue_vehicles"] == 5
    assert flat["north_queue_estimate_cm"] == 0
    assert flat["south_queue_vehicles"] == 1
    assert "camera" not in str(type(flat.get("north_queue_vehicles")))


def test_topics():
    t = mqtt.canonical_topics("SIMPANG_TALUN_01")
    assert t["vision"].endswith("/vision/metrics")
    assert t["recommendation"].endswith("/control/recommendation")
    assert "controllers/+/telemetry" in t["telemetry"]
