"""
vision_fuzzy_server.py
========================================================================================
FastAPI Vision AI & Fuzzy Logic Adaptive Traffic Control Server
========================================================================================
Endpoints:
- GET  /health           : Health check & model status
- POST /detect           : Ingest camera frame (ESP32-CAM / CCTV), detect, update controller
- GET  /status           : Real-time traffic signal state, SMP loads, & timer
- POST /control/emergency: Emergency preemption override trigger
- WS   /ws               : Real-time WebSocket feed for Web Dashboard / Vision Lab
========================================================================================
"""

from __future__ import annotations

import os
import io
import json
import time
import asyncio
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from adaptive_traffic_engine import (
    VisionTrafficDetector,
    MultiLaneTrafficController,
    DEFAULT_LANES,
)

# ======================================================================================
# KONFIGURASI ENVIRONMENT
# ======================================================================================
MODEL_PATH = os.getenv("DETECT_MODEL", "yolov8n.pt")
CONF_THRESHOLD = float(os.getenv("DETECT_CONF", "0.25"))
IOU_THRESHOLD = float(os.getenv("DETECT_IOU", "0.60"))
IMGSZ = int(os.getenv("DETECT_IMGSZ", "960"))
MIN_GREEN = int(os.getenv("MIN_GREEN_SEC", "10"))
MAX_GREEN = int(os.getenv("MAX_GREEN_SEC", "60"))
MQTT_HOST = os.getenv("MQTT_HOST", None)
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))

# Inisialisasi Vision Detector & Multi-Lane Controller
detector = VisionTrafficDetector(
    model_path=MODEL_PATH,
    conf_thresh=CONF_THRESHOLD,
    iou_thresh=IOU_THRESHOLD,
    imgsz=IMGSZ,
)

controller = MultiLaneTrafficController(
    lane_names=DEFAULT_LANES,
    min_green=MIN_GREEN,
    max_green=MAX_GREEN,
)

# Background Ticker Task
async def controller_tick_loop():
    """Background task untuk memajukan state machine pengatur lampu tiap detik."""
    while True:
        try:
            controller.tick()
        except Exception as e:
            print(f"[Controller Loop Error]: {e}")
        await asyncio.sleep(1.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: jalankan background controller tick
    task = asyncio.create_task(controller_tick_loop())
    print("=" * 60)
    print(f"  Adaptive Traffic Vision & Fuzzy Server Online")
    print(f"  Model       : {MODEL_PATH}")
    print(f"  Conf Thresh : {CONF_THRESHOLD} | ImgSz: {IMGSZ}")
    print(f"  Green Range : {MIN_GREEN}s - {MAX_GREEN}s")
    print("=" * 60)
    yield
    # Shutdown
    task.cancel()


app = FastAPI(
    title="Adaptive Traffic Vision & Fuzzy Control Server",
    description="Vision AI (YOLO) + MKJI SMP Calculation + Fuzzy Logic Traffic Signal System",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ======================================================================================
# MODELS & SCHEMAS
# ======================================================================================
class EmergencyRequest(BaseModel):
    lane: str
    active: bool = True


# ======================================================================================
# HTTP ENDPOINTS
# ======================================================================================
@app.get("/")
def index():
    return {
        "status": "online",
        "service": "Adaptive Traffic Vision & Fuzzy Control Server",
        "docs": "/docs",
        "health": "/health",
        "status_endpoint": "/status",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": detector._model is not None,
        "model_path": MODEL_PATH,
        "conf_threshold": CONF_THRESHOLD,
        "imgsz": IMGSZ,
        "lanes": DEFAULT_LANES,
    }


@app.post("/detect")
async def detect_frame(
    frame: UploadFile = File(...),
    lane: str = Form(default="north"),
):
    """
    Menerima frame gambar JPEG dari ESP32-CAM atau CCTV,
    menjalankan deteksi YOLO + Rider Fusion, menghitung beban SMP,
    dan memperbarui status kontroler lalu lintas.
    """
    lane_key = lane.lower()
    if lane_key not in DEFAULT_LANES:
        lane_key = "north"

    t0 = time.time()
    contents = await frame.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image payload")

    # Jalankan deteksi
    det_result = detector.detect_frame(img)
    latency_ms = round((time.time() - t0) * 1000, 1)

    # Perbarui data pada controller
    controller.update_lane_detection(
        lane_name=lane_key,
        counts=det_result["counts"],
        emergency=det_result["emergency"],
    )

    current_status = controller.get_status()

    return {
        "lane": lane_key,
        "inference_ms": latency_ms,
        "total_vehicles": sum(det_result["counts"].values()),
        "counts": det_result["counts"],
        "smp_load": det_result["smp_load"],
        "emergency_detected": det_result["emergency"],
        "boxes": det_result["boxes"],
        "classes": det_result["classes"],
        "confs": det_result["confs"],
        "current_signal": current_status["lanes"].get(lane_key, {}).get("signal", "RED"),
        "active_green_lane": current_status["active_green_lane"],
        "time_remaining_seconds": current_status["time_remaining_seconds"],
    }


@app.get("/status")
def get_traffic_status():
    """Mengembalikan status terkini seluruh persimpangan & timer lampu hijau."""
    return controller.get_status()


@app.post("/control/emergency")
def trigger_emergency(req: EmergencyRequest):
    """Memicu atau membatalkan override darurat untuk lajur tertentu."""
    lane_key = req.lane.lower()
    if lane_key not in DEFAULT_LANES:
        raise HTTPException(status_code=400, detail=f"Invalid lane. Must be one of {DEFAULT_LANES}")

    controller.lanes[lane_key].emergency_active = req.active
    return {
        "success": True,
        "lane": lane_key,
        "emergency_active": req.active,
        "current_status": controller.get_status(),
    }


# ======================================================================================
# WEBSOCKET STREAMING ENDPOINT (Untuk Web Dashboard)
# ======================================================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[WebSocket] Client connected: {websocket.client}")
    try:
        while True:
            # Terima pesan (baik binary JPEG frame maupun JSON)
            data = await websocket.receive()
            t0 = time.time()

            if "bytes" in data and data["bytes"]:
                nparr = np.frombuffer(data["bytes"], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                lane_key = "north"
            elif "text" in data and data["text"]:
                try:
                    payload = json.loads(data["text"])
                    lane_key = payload.get("lane", "north")
                    if "image_base64" in payload:
                        import base64
                        raw_bytes = base64.b64decode(payload["image_base64"])
                        nparr = np.frombuffer(raw_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    else:
                        img = None
                except Exception:
                    img = None
                    lane_key = "north"
            else:
                img = None
                lane_key = "north"

            if img is not None:
                det_result = detector.detect_frame(img)
                controller.update_lane_detection(
                    lane_name=lane_key,
                    counts=det_result["counts"],
                    emergency=det_result["emergency"],
                )
            else:
                det_result = {
                    "counts": {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0},
                    "smp_load": 0.0,
                    "boxes": [],
                    "classes": [],
                    "emergency": False,
                }

            latency_ms = round((time.time() - t0) * 1000, 1)
            status = controller.get_status()

            response = {
                "type": "telemetry",
                "inference_ms": latency_ms,
                "detection": det_result,
                "traffic_status": status,
            }
            await websocket.send_text(json.dumps(response))

    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket Error]: {e}")


# ======================================================================================
# STANDALONE RUNNER
# ======================================================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8081"))
    uvicorn.run("vision_fuzzy_server:app", host="0.0.0.0", port=port, reload=False)
