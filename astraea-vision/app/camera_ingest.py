"""Ingest WSS kamera: auth dulu, lalu frame JPEG biner (PRD §9/CAM-FW-03, §10).

Protokol:
  1. pesan pertama = JSON teks {"camera_id","intersection_id","approach_id","token","firmware"}
  2. balasan {"ok":true} atau tutup koneksi (kode 4401/4403/4404)
  3. frame = pesan biner JPEG (profil: VGA, q12-18, 2-5 FPS, latest-only §CAM-FW-04)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import cv2
import numpy as np
from websockets.asyncio.server import serve

from . import auth as camera_auth
from . import camera_registry, config
from .frame_store import STORE
from .metrics import HUB

logger = logging.getLogger("vision.ingest")

AUTH_TIMEOUT_S = 10.0


async def handle_camera(ws) -> None:
    peer = getattr(ws, "remote_address", "?")
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=AUTH_TIMEOUT_S)
    except (asyncio.TimeoutError, Exception):
        await ws.close(code=4408, reason="auth-timeout")
        return
    try:
        hello = json.loads(raw if isinstance(raw, str) else raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        await ws.close(code=4400, reason="bad-hello")
        return
    camera_id = str(hello.get("camera_id", ""))
    intersection_id = str(hello.get("intersection_id", ""))
    approach_id = str(hello.get("approach_id", ""))
    token = str(hello.get("token", ""))
    firmware = str(hello.get("firmware", ""))
    ok, reason, cam = camera_auth.verify_camera(camera_id, intersection_id, approach_id, token)
    if not ok:
        code = {"unknown-camera": 4404, "camera-disabled": 4403}.get(reason, 4401)
        logger.warning("ingest-reject %s (%s) reason=%s", camera_id or peer, peer, reason)
        # F9: kirim JSON error DULU agar firmware mudah diagnosis, baru close.
        try:
            await ws.send(json.dumps({"ok": False, "error": reason}))
        except Exception:
            pass
        await ws.close(code=code, reason=reason)
        return
    logger.info("ingest-accept camera=%s approach=%s fw=%s", camera_id, approach_id, firmware)
    HUB.register(camera_id, intersection_id, approach_id)
    HUB.set_connected(camera_id, True)
    frames = 0
    dropped = 0
    t0 = time.monotonic()
    first_frame_at: float | None = None
    last_seen_push = t0
    try:
        await ws.send(json.dumps({"ok": True, "server": "astraea-vision"}))
        async for msg in ws:
            if isinstance(msg, str):
                continue  # abaikan teks non-hello
            arr = np.frombuffer(msg, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                dropped += 1
                continue
            h, w = img.shape[:2]
            STORE.put(camera_id, bytes(msg), width=w, height=h)
            frames += 1
            if first_frame_at is None:
                first_frame_at = time.monotonic()
            now = time.monotonic()
            if now - last_seen_push > 5.0 and first_frame_at is not None:
                fps = frames / max(1.0, now - first_frame_at)
                try:
                    camera_registry.touch_seen(camera_id, fps_ingest=round(fps, 1))
                except Exception as exc:
                    logger.warning("touch_seen failed: %s", exc)
                last_seen_push = now
    except Exception as exc:
        logger.info("ingest-closed camera=%s: %s", camera_id, exc)
    finally:
        HUB.set_connected(camera_id, False)
        logger.info("ingest-end camera=%s frames=%d dropped=%d", camera_id, frames, dropped)


async def run_forever() -> None:
    logger.info("ingest listening ws://%s:%d%s", config.WS_HOST, config.WS_PORT, config.WS_PATH)
    async with serve(
        handle_camera, config.WS_HOST, config.WS_PORT,
        max_size=config.MAX_FRAME_BYTES, process_request=_route_check,
    ):
        await asyncio.get_running_loop().create_future()


async def _route_check(connection, request):
    if request.path != config.WS_PATH:
        return connection.respond(404, "not found\n")
    return None
