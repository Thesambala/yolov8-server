#!/usr/bin/env python3
"""
ASTRAEA MQTT Bridge (EC2 #2 -> EC2 #1)
Setiap PUBLISH_INTERVAL detik, baca agregat deteksi terakhir dari
~/data/runs/latest_stats.json (ditulis oleh yolov8-vision-server/server.py)
lalu publish ke broker MQTT di EC2 #1.

Topic   : traffic/CAM_YOLO_01/data
Payload : {"device_id", "timestamp",
           "north_vehicle_count", "north_density_level",
           "south_vehicle_count", "south_density_level",
           "east_vehicle_count",  "east_density_level"}
Density : 0-3 kendaraan = level 0, 4-7 = level 1, >=8 = level 2

Konfigurasi via environment (jangan hardcode kredensial):
  MQTT_HOST, MQTT_PORT=1883, MQTT_USER, MQTT_PASS,
  DEVICE_ID=CAM_YOLO_01, PUBLISH_INTERVAL=5
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astraea-mqtt-bridge")

MQTT_HOST = os.environ.get("MQTT_HOST", "")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")
DEVICE_ID = os.environ.get("DEVICE_ID", "CAM_YOLO_01")
INTERVAL = float(os.environ.get("PUBLISH_INTERVAL", "5"))

STATS_FILE = "/home/ubuntu/data/runs/latest_stats.json"
STALE_AFTER = 15  # detik; data lebih tua dari ini dianggap basi, tidak dipublish


def density_level(count: int) -> int:
    if count <= 3:
        return 0
    if count <= 7:
        return 1
    return 2


def load_stats() -> dict | None:
    try:
        with open(STATS_FILE) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.warning(f"{STATS_FILE} belum ada — belum ada deteksi yang masuk")
        return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Gagal membaca {STATS_FILE}: {e}")
        return None

    age = time.time() - data.get("timestamp", 0)
    if age > STALE_AFTER:
        logger.info(f"Data basi ({age:.0f}s) — skip publish")
        return None
    return data


def build_payload(data: dict) -> dict:
    n, s, e = data.get("north", 0), data.get("south", 0), data.get("east", 0)
    ts = datetime.fromtimestamp(data.get("timestamp", time.time()), tz=timezone.utc)
    def q_est(c): return 0 if c <=3 else 20 if c <=7 else 40
    return {
        "device_id": DEVICE_ID,
        "timestamp": ts.isoformat(),
        "sensor_mode": True,
        "dummy_mode": False,
        "auto_mode": True,
        "adaptive_mode": True,
        "north_vehicle_count": n,
        "north_density_level": density_level(n),
        "north_queue_detected": density_level(n) >=2,
        "north_queue_estimate_cm": q_est(n),
        "north_queue_vehicles": n,
        "south_vehicle_count": s,
        "south_density_level": density_level(s),
        "south_queue_detected": density_level(s) >=2,
        "south_queue_estimate_cm": q_est(s),
        "south_queue_vehicles": s,
        "east_vehicle_count": e,
        "east_density_level": density_level(e),
        "east_queue_detected": density_level(e) >=2,
        "east_queue_estimate_cm": q_est(e),
        "east_queue_vehicles": e,
    }


def main():
    if not MQTT_HOST or not MQTT_USER:
        logger.error("MQTT_HOST / MQTT_USER belum diisi (env file: ~/.config/astraea/mqtt.env)")
        raise SystemExit(1)

    topic = f"traffic/{DEVICE_ID}/data"
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"yolo-bridge-{DEVICE_ID}",
    )
    if MQTT_USER:
        client.username_pw_set(MQTT_USER, MQTT_PASS)

    connected = False

    def on_connect(c, userdata, flags, reason_code, properties):
        nonlocal connected
        if reason_code == 0:
            connected = True
            logger.info(f"Terhubung ke {MQTT_HOST}:{MQTT_PORT} — publishing ke {topic} tiap {INTERVAL:.0f}s")
        else:
            logger.error(f"Connect gagal: {reason_code}")

    def on_disconnect(c, userdata, flags, reason_code, properties):
        nonlocal connected
        connected = False
        logger.warning(f"Terputus dari broker (reason={reason_code})")

    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.connect(MQTT_HOST, MQTT_PORT, keepalive=30)
    client.loop_start()

    try:
        while True:
            if connected:
                data = load_stats()
                if data is not None:
                    payload = build_payload(data)
                    info = client.publish(topic, json.dumps(payload), qos=0, retain=False)
                    if info.rc == mqtt.MQTT_ERR_SUCCESS:
                        logger.info(
                            f"-> {topic} N={payload['north_vehicle_count']}(L{payload['north_density_level']}) "
                            f"S={payload['south_vehicle_count']}(L{payload['south_density_level']}) "
                            f"E={payload['east_vehicle_count']}(L{payload['east_density_level']})"
                        )
                    else:
                        logger.warning(f"Publish gagal rc={info.rc}")
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        pass
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
