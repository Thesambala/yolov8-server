"""Annotator murni: gambar track ke frame. TIDAK menjalankan YOLO.

Input = tracks dari TrackerHub.track_frame (satu hasil inferensi).
Output = image OpenCV teranotasi. Gagal gambar/encode tidak boleh
mengganggu metrik/fuzzy/MQTT (ditangani pemanggil).
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Set

import cv2
import numpy as np

JPEG_QUALITY = int(os.getenv("ANNOTATED_JPEG_QUALITY", "85"))
DEBUG = os.getenv("ANNOTATION_DEBUG", "false").lower() in ("1", "true", "yes")

# Biru = jalan, oranye = berhenti (konsisten, tidak berlebihan).
COLOR_MOVE = (255, 120, 40)
COLOR_STOP = (40, 140, 255)


def _box_px(bbox: Dict[str, Any], w: int, h: int):
    try:
        x1 = int(float(bbox.get("x1", 0)) * w)
        y1 = int(float(bbox.get("y1", 0)) * h)
        x2 = int(float(bbox.get("x2", 0)) * w)
        y2 = int(float(bbox.get("y2", 0)) * h)
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, x1), min(w - 1, x2)))
    y1, y2 = sorted((max(0, y1), min(h - 1, y2)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def annotate_frame(
    frame: np.ndarray,
    tracks: List[Dict[str, Any]],
    stopped_ids: Set[int] | List[int] | None = None,
    count_line_y: float | None = None,
    camera_id: str = "",
    source_seq: int = 0,
    latency_s: float = 0.0,
) -> np.ndarray:
    """Gambar bbox+label ke SALINAN frame; frame asli tak diubah."""
    stopped = set(stopped_ids or [])
    out = frame.copy()
    h, w = out.shape[:2]
    for tr in tracks:
        try:
            tid = int(tr.get("track_id", -1))
        except (TypeError, ValueError):
            continue
        box = _box_px(tr.get("bbox") or {}, w, h)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        is_stop = tid in stopped
        color = COLOR_STOP if is_stop else COLOR_MOVE
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = str(tr.get("label", ""))
        try:
            conf = float(tr.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        text = f"{label} {conf:.0%}" if label else f"{conf:.0%}"
        text += f" #{tid}" if tid >= 0 else ""
        if is_stop:
            text += " STOPPED"
        y0 = max(0, y1 - 6)
        cv2.putText(out, text[:48], (x1, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                    cv2.LINE_AA)
    if count_line_y is not None:
        try:
            ly = int(float(count_line_y) / 100.0 * h)
            cv2.line(out, (0, ly), (w, ly), (140, 140, 140), 1)
            cv2.putText(out, "COUNT LINE", (6, max(0, ly - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
        except (TypeError, ValueError):
            pass
    if DEBUG:
        cv2.putText(out, f"{camera_id} seq={source_seq} lat={latency_s:.2f}s",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
                    cv2.LINE_AA)
    return out


def encode_jpeg(frame: np.ndarray, quality: int = JPEG_QUALITY) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return bytes(buf) if ok else None
