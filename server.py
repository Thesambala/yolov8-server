#!/usr/bin/env python3
"""
ASTRAEA YOLOv8 WebSocket Inference Server
Port 8080, path /ws — dipakai halaman Vision Lab (app/vision-lab-7x9k-alpha).

Protokol (pola docs/YOLOV8_SETUP.md), per pesan:
  - binary            : frame JPEG mentah
  - JSON {"type":"frame","data":"<base64>"} : frame JPEG base64 (dengan/tanpa prefix data:)
  - teks base64 mentah                     : fallback, dianggap frame
  - JSON {"type":"config","confidence":0.5}: ubah threshold confidence

Balasan: {"type":"detections","detections":[...],"stats":{...}} — field
detections/stats identik dengan endpoint HTTP /detect (box dinormalisasi 0-100).

Run:
  /home/ubuntu/yolo-env/bin/python /home/ubuntu/yolov8-vision-server/server.py
"""

import asyncio
import base64
import json
import logging
import os
import tempfile
import time

import cv2
import numpy as np
from ultralytics import YOLO
from websockets.asyncio.server import serve

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("astraea-ws")

MODEL_PATH = os.getenv("DETECT_MODEL", "/home/ubuntu/models/yolov8n.pt")
CONFIDENCE = float(os.getenv("DETECT_CONF", "0.30"))  # default per-koneksi
INFER_IMGSZ = int(os.getenv("DETECT_IMGSZ", "1280"))  # fallback 960 kalau lambat
# Frame dari browser sering di-downscale (mis. 640px) — upscale dulu agar objek
# kecil tidak hilang. 0/kosong = off.
MIN_SRC_WIDTH = int(os.getenv("DETECT_MIN_SRC_WIDTH", "960"))
# ROI opsional "x,y,w,h" relatif 0-100: crop area jalan saja sebelum inferensi.
# Kosong = pakai frame penuh. Identik dengan server_detect_http.py.
_roi_env = os.getenv("DETECT_ROI", "").strip()
ROI = tuple(float(v) for v in _roi_env.split(",") if v != "") if _roi_env else None
# Isi path folder utk menyimpan frame yang benar-benar diterima (diagnosa browser)
DEBUG_DIR = os.getenv("YOLO_DEBUG_FRAMES", "").strip()
HOST = "0.0.0.0"
PORT = 8080
MAX_FRAME_BYTES = 16 * 1024 * 1024  # batas ukuran pesan WS

# Mapping kelas model -> label. Default: ID COCO. Untuk model hasil fine-tune,
# set env DETECT_CLASSES="0:car,1:motorcycle,2:bus,3:truck" (urutan = data.yaml).
DEFAULT_CLASSES = "2:car,5:bus,7:truck,3:motorcycle,1:bicycle"
_classes_env = os.getenv("DETECT_CLASSES", DEFAULT_CLASSES).strip()
try:
    VEHICLE_CLASSES = {
        int(k.strip()): v.strip()
        for k, v in (pair.split(":", 1) for pair in _classes_env.split(",") if pair)
    }
except ValueError:
    logger.warning(f"DETECT_CLASSES invalid: {_classes_env}, pakai default COCO")
    VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck", 3: "motorcycle", 1: "bicycle"}

logger.info(f"Loading model {MODEL_PATH}...")
logger.info(f"Config: conf={CONFIDENCE} imgsz={INFER_IMGSZ} roi={ROI} "
            f"min_src_width={MIN_SRC_WIDTH}")
model = YOLO(MODEL_PATH)
logger.info("Model loaded.")

# FPS stats
frame_count = 0
last_fps_time = time.time()
fps = 0.0

# File state untuk mqtt_bridge.py (agregat per arah, dipartisi berdasar posisi
# tengah bounding box: atas=north, bawah=south, tengah=east — sesuaikan bila
# geometri lane diketahui)
STATS_FILE = "/home/ubuntu/data/runs/latest_stats.json"


def write_stats_file(stats: dict):
    """Simpan agregat terakhir secara atomik untuk dibaca mqtt_bridge."""
    try:
        payload = {
            "timestamp": time.time(),
            "totalVehicles": stats.get("totalVehicles", 0),
            "north": stats.get("_north", 0),
            "south": stats.get("_south", 0),
            "east": stats.get("_east", 0),
        }
        d = os.path.dirname(STATS_FILE)
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, STATS_FILE)
    except Exception as e:
        logger.warning(f"Gagal menulis stats file: {e}")


def apply_roi(img: np.ndarray):
    """Crop ROI relatif 0-100. Return (crop, offset_x_px, offset_y_px)."""
    if not ROI or len(ROI) != 4:
        return img, 0, 0
    h, w = img.shape[:2]
    rx, ry, rw, rh = ROI
    x1 = max(0, int(rx / 100 * w))
    y1 = max(0, int(ry / 100 * h))
    x2 = min(w, int((rx + rw) / 100 * w))
    y2 = min(h, int((ry + rh) / 100 * h))
    if x2 - x1 < 32 or y2 - y1 < 32:
        logger.warning(f"ROI terlalu kecil ({x1},{y1},{x2},{y2}), pakai frame penuh")
        return img, 0, 0
    return img[y1:y2, x1:x2], x1, y1


def prepare_infer(img: np.ndarray):
    """ROI crop + upscale frame kecil. Return (img_inf, ox, oy, sx, sy) —
    koordinat asli = ox + x_inf*sx , oy + y_inf*sy."""
    crop, ox, oy = apply_roi(img)
    ch, cw = crop.shape[:2]
    sx = sy = 1.0
    if MIN_SRC_WIDTH > 0 and cw < MIN_SRC_WIDTH:
        f = MIN_SRC_WIDTH / cw
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
        sx = sy = 1.0 / f
    return crop, ox, oy, sx, sy


def infer(img: np.ndarray, conf_thres: float | None = None) -> dict:
    """YOLOv8 inference -> dict deteksi (format sama dengan /detect)."""
    global frame_count, last_fps_time, fps

    h, w = img.shape[:2]
    crop, ox, oy, sx, sy = prepare_infer(img)
    t0 = time.time()
    results = model(crop, conf=conf_thres or CONFIDENCE,
                    imgsz=INFER_IMGSZ, verbose=False)
    inference_ms = (time.time() - t0) * 1000

    detections = []
    stats = {
        "car": 0, "truck": 0, "bus": 0,
        "motorcycle": 0, "bicycle": 0,
        "totalVehicles": 0,
        "inferenceMs": round(inference_ms, 1),
    }

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue
            label = VEHICLE_CLASSES[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # kembalikan ke skala frame penuh (offset ROI + skala upscale)
            px, py = ox + x1 * sx, oy + y1 * sy
            bw, bh = (x2 - x1) * sx, (y2 - y1) * sy
            detections.append({
                "label": label,
                "confidence": round(conf, 3),
                # koordinat dinormalisasi 0-100 agar mudah digambar overlay
                "x": round(px / w * 100, 1),
                "y": round(py / h * 100, 1),
                "w": round(bw / w * 100, 1),
                "h": round(bh / h * 100, 1),
            })
            stats[label] += 1
            stats["totalVehicles"] += 1
            # partisi arah berdasar posisi tengah box (atas=north, bawah=south)
            cy_pct = (py + bh / 2) / h * 100
            if cy_pct < 33.4:
                stats["_north"] = stats.get("_north", 0) + 1
            elif cy_pct > 66.7:
                stats["_south"] = stats.get("_south", 0) + 1
            else:
                stats["_east"] = stats.get("_east", 0) + 1

    write_stats_file(stats)
    for k in ("_north", "_south", "_east"):
        stats.pop(k, None)

    frame_count += 1
    now = time.time()
    if now - last_fps_time >= 1.0:
        fps = frame_count / (now - last_fps_time)
        frame_count = 0
        last_fps_time = now
    stats["fps"] = round(fps, 1)

    return {"type": "detections", "detections": detections, "stats": stats}


def decode_jpeg(payload: bytes):
    """bytes JPEG -> image BGR (None jika tidak valid)."""
    arr = np.frombuffer(payload, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


_debug_saved = 0


async def handle_frame(connection, jpeg_bytes: bytes, state: dict):
    global _debug_saved
    img = await asyncio.get_running_loop().run_in_executor(None, decode_jpeg, jpeg_bytes)
    if img is None:
        await connection.send(json.dumps({"type": "error", "error": "bukan JPEG valid"}))
        return
    fh, fw = img.shape[:2]
    if not state.get("logged_size"):
        logger.info(f"Frame pertama dari {connection.remote_address}: {fw}x{fh}px")
        state["logged_size"] = True
    if DEBUG_DIR:
        try:
            os.makedirs(DEBUG_DIR, exist_ok=True)
            if _debug_saved < 10:
                cv2.imwrite(os.path.join(
                    DEBUG_DIR,
                    f"recv_{int(time.time())}_{fw}x{fh}.jpg"), img)
                _debug_saved += 1
        except Exception as e:
            logger.warning(f"Gagal simpan debug frame: {e}")
    result = await asyncio.get_running_loop().run_in_executor(
        None, infer, img, state["conf"])
    await connection.send(json.dumps(result))


async def handler(connection):
    peer = connection.remote_address
    path = connection.request.path if connection.request else "/"
    # conf per-koneksi: slider browser TIDAK lagi menimpa klien lain
    state = {"conf": CONFIDENCE, "logged_size": False}
    logger.info(f"Client connected: {peer} (path={path}) conf={state['conf']}")
    try:
        async for message in connection:
            if isinstance(message, bytes):
                await handle_frame(connection, message, state)
                continue
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                # mungkin raw base64 frame
                b64 = message.split(",")[1] if "," in message else message
                try:
                    jpeg = base64.b64decode(b64)
                except Exception:
                    await connection.send(json.dumps({"type": "error", "error": "pesan tidak dikenal"}))
                    continue
                await handle_frame(connection, jpeg, state)
                continue

            mtype = data.get("type")
            if mtype == "frame" and "data" in data:
                b64 = data["data"].split(",")[1] if "," in data["data"] else data["data"]
                try:
                    jpeg = base64.b64decode(b64)
                except Exception as e:
                    await connection.send(json.dumps({"type": "error", "error": f"base64 invalid: {e}"}))
                    continue
                await handle_frame(connection, jpeg, state)
            elif mtype == "config":
                if "confidence" in data:
                    state["conf"] = max(0.05, min(0.9, float(data["confidence"])))
                    logger.info(f"Confidence (per-koneksi {peer}) -> {state['conf']}")
                await connection.send(json.dumps({"type": "config", "confidence": state["conf"]}))
            else:
                await connection.send(json.dumps({"type": "error", "error": "unknown message type"}))
    except Exception as e:
        logger.warning(f"Connection error {peer}: {e}")
    finally:
        logger.info(f"Client left: {peer}")


async def main():
    async with serve(handler, HOST, PORT, max_size=MAX_FRAME_BYTES,
                     ping_interval=20, ping_timeout=20):
        logger.info(f"YOLOv8 Vision Server running on ws://{HOST}:{PORT}/ws")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
