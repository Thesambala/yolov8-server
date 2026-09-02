#!/usr/bin/env python3
"""
ASTRAEA YOLOv8 HTTP Detect Endpoint (JALUR B - headless ESP32-CAM)
Port 8081 — menerima POST multipart field "frame" (JPEG),
balas JSON deteksi. Jalankan berdampingan dengan server.py (WS, port 8080).

Install:
  source venv/bin/activate
  pip install fastapi uvicorn python-multipart

Run:
  uvicorn server_detect_http:app --host 0.0.0.0 --port 8081

Test dari laptop:
  curl -F "frame=@test.jpg" http://<EC2-IP>:8081/detect
"""

import os
import time
import logging
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("astraea-detect")

MODEL_NAME = os.getenv("DETECT_MODEL", "/home/ubuntu/models/yolov8n.pt")
CONFIDENCE = float(os.getenv("DETECT_CONF", "0.30"))
INFER_IMGSZ = int(os.getenv("DETECT_IMGSZ", "1280"))  # fallback 960 kalau lambat
# Frame kecil (ESP32-CAM/browser) di-upscale dulu agar objek kecil tidak hilang
MIN_SRC_WIDTH = int(os.getenv("DETECT_MIN_SRC_WIDTH", "960"))
# ROI opsional "x,y,w,h" relatif 0-100: crop area jalan saja sebelum inferensi
# (objek jadi relatif lebih besar, false positive turun). Kosong = pakai frame penuh.
_roi_env = os.getenv("DETECT_ROI", "").strip()
ROI = tuple(float(v) for v in _roi_env.split(",") if v != "") if _roi_env else None
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

logger.info(f"Config: conf={CONFIDENCE} imgsz={INFER_IMGSZ} roi={ROI} "
            f"min_src_width={MIN_SRC_WIDTH}")

model = YOLO(MODEL_NAME)
app = FastAPI(title="ASTRAEA Detect API")


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


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/detect")
async def detect(frame: UploadFile = File(...)):
    t0 = time.time()

    data = await frame.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "bukan JPEG valid"}

    h, w = img.shape[:2]
    crop, ox, oy, sx, sy = prepare_infer(img)
    results = model(crop, conf=CONFIDENCE, imgsz=INFER_IMGSZ, verbose=False)

    detections: List[dict] = []
    stats = {
        "car": 0, "truck": 0, "bus": 0,
        "motorcycle": 0, "bicycle": 0,
        "totalVehicles": 0,
        "inferenceMs": round((time.time() - t0) * 1000, 1),
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

    return {"detections": detections, "stats": stats}
