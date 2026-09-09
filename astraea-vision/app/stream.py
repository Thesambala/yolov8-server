"""HTTP API: health, status/snapshot/stream kamera, rekomendasi, deteksi-kompatibel (PRD §10/42)."""
from __future__ import annotations

import asyncio
import logging
import time

import cv2
import numpy as np
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse

from . import camera_registry, config
from .frame_store import ANNOTATED_STORE, STORE
from .metrics import HUB

logger = logging.getLogger("vision.http")

MODEL_INFO: dict = {}
STARTED_AT = time.time()


def _viewer_ok(authorization: str | None) -> bool:
    if not config.VIEWER_TOKEN:
        return True  # tanpa token = mode dev; produksi WAJIB set
    return authorization == f"Bearer {config.VIEWER_TOKEN}"


app = FastAPI(title="astraea-vision-service")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "astraea-vision-service",
        "uptime_s": round(time.time() - STARTED_AT, 1),
        "model": MODEL_INFO,
        "cameras": STORE.camera_ids(),
    }


@app.get("/v1/cameras/{camera_id}/status")
def camera_status(camera_id: str, authorization: str | None = Header(default=None)):
    if not _viewer_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        reg = camera_registry.get_camera(camera_id)
    except Exception:
        reg = None
    if not reg:
        raise HTTPException(status_code=404, detail="unknown camera")
    st = HUB.camera_state(camera_id)
    try:
        frame_age = STORE.age_s(camera_id)
    except Exception:
        frame_age = None
    ann = ANNOTATED_STORE.get(camera_id)
    ann_age = ANNOTATED_STORE.age_s(camera_id)
    return {
        "camera_id": camera_id,
        "intersection_id": reg.get("intersection_id"),
        "approach_id": reg.get("approach_id"),
        "enabled": reg.get("enabled"),
        "health_state": reg.get("health_state"),
        "last_seen": reg.get("last_seen"),
        "fps_ingest": reg.get("fps_ingest"),
        "frame_age_s": round(frame_age, 1) if frame_age is not None else None,
        "annotated_available": ann is not None,
        "annotated_age_s": round(ann_age, 1) if ann_age is not None else None,
        "annotated_source_seq": ann.get("source_seq") if ann else None,
        "metrics": st,
    }


@app.get("/v1/cameras/{camera_id}/snapshot.jpg")
def snapshot(camera_id: str, authorization: str | None = Header(default=None)):
    if not _viewer_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    item = STORE.get(camera_id)
    if not item:
        raise HTTPException(status_code=404, detail="no frame yet")
    return Response(content=item["jpeg"], media_type="image/jpeg")


@app.get("/v1/cameras/{camera_id}/annotated.jpg")
def annotated(camera_id: str, authorization: str | None = Header(default=None)):
    if not _viewer_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    try:
        reg = camera_registry.get_camera(camera_id)
    except Exception:
        reg = None
    if not reg:
        raise HTTPException(status_code=404, detail="unknown camera")
    item = ANNOTATED_STORE.get(camera_id)
    if not item:
        raise HTTPException(status_code=404, detail="no annotated frame yet")
    return Response(content=item["jpeg"], media_type="image/jpeg")


@app.get("/v1/cameras/{camera_id}/stream.mjpeg")
def stream_mjpeg(camera_id: str, authorization: str | None = Header(default=None)):
    if not _viewer_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")

    async def gen():
        boundary = b"--frame"
        while True:
            item = STORE.get(camera_id)
            if item:
                yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + item["jpeg"] + b"\r\n"
            await asyncio.sleep(0.4)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/v1/control/recommendation/{intersection_id}")
def recommendation_view(intersection_id: str,
                        authorization: str | None = Header(default=None)):
    if not _viewer_ok(authorization):
        raise HTTPException(status_code=401, detail="unauthorized")
    out = {}
    for approach in HUB.approaches_of(intersection_id):
        out[approach] = HUB.approach_state(intersection_id, approach)
    return {"intersection_id": intersection_id, "approaches": out}


# Kompatibilitas dengan firmware JALUR B lama (ESP32-CAM POST /detect).
@app.post("/detect")
async def detect_compat(frame: UploadFile = File(...)):
    from .yolo_singleton import ENGINE  # impor malas: butuh model

    if ENGINE.model is None:
        info = ENGINE.load()
        MODEL_INFO.update(info)
    raw = await frame.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(status_code=400, content={"detail": "invalid jpeg"})
    dets, stats = ENGINE.detect(img)
    return {"type": "detections", "detections": dets, "stats": stats}
