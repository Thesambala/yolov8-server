"""YOLO engine: load model + diagnostik startup (PRD §31) + inferensi per kamera."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from . import config

logger = logging.getLogger("vision.yolo")

# Nama kelas model kustom -> kategori generik. Dibaca dari metadata model saat startup;
# tidak ada mapping nomor COCO yang di-hardcode (PRD §11.1).
PERSON_HINTS = ("pejalan", "pedestrian", "person", "people", "orang")
NON_MOTOR_HINTS = ("unmotorized", "sepeda", "bicycle", "becak")


class YoloEngine:
    def __init__(self) -> None:
        self.model: YOLO | None = None
        self.class_names: Dict[int, str] = {}
        self.device = "cpu"

    def load(self) -> Dict[str, Any]:
        t0 = time.monotonic()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(config.MODEL_PATH)
        try:
            names = self.model.names or {}
        except Exception:
            names = {}
        self.class_names = {int(k): str(v) for k, v in dict(names).items()}
        info: Dict[str, Any] = {
            "model_path": config.MODEL_PATH,
            "architecture": type(getattr(self.model, "model", None)).__name__,
            "task": getattr(self.model, "task", "detect"),
            "classes": self.class_names,
            "n_classes": len(self.class_names),
            "imgsz": config.INFER_IMGSZ,
            "conf": config.CONFIDENCE,
            "iou": config.IOU,
            "device": self.device,
        }
        # Warm-up agar alokasi pertama tidak dihitung sebagai latensi inferensi.
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        w0 = time.monotonic()
        try:
            self.model.predict(dummy, imgsz=320, conf=0.9, verbose=False)
            info["warmup_s"] = round(time.monotonic() - w0, 3)
            info["warmup"] = "ok"
        except Exception as exc:
            info["warmup"] = f"failed: {exc}"
        info["load_s"] = round(time.monotonic() - t0, 3)
        logger.info("Model loaded: %s", {k: v for k, v in info.items() if k != "classes"})
        logger.info("Model classes: %s", self.class_names)
        return info

    def category_of(self, class_id: int) -> str:
        name = self.class_names.get(int(class_id), "").lower()
        if any(h in name for h in PERSON_HINTS):
            return "person"
        if any(h in name for h in NON_MOTOR_HINTS):
            return "unmotorized"
        return "vehicle"

    @staticmethod
    def upscale_small(frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if 0 < w < config.MIN_SRC_WIDTH:
            scale = config.MIN_SRC_WIDTH / float(w)
            frame = cv2.resize(frame, (config.MIN_SRC_WIDTH, int(h * scale)))
        return frame

    @staticmethod
    def crop_roi(frame: np.ndarray, roi: Tuple[float, float, float, float] | None):
        """ROI relatif 0-100 'x,y,w,h'. Return (crop, (ox, oy, scale_info))."""
        if not roi:
            return frame, (0, 0)
        h, w = frame.shape[:2]
        x, y, rw, rh = roi
        x0, y0 = int(w * x / 100.0), int(h * y / 100.0)
        x1, y1 = int(w * (x + rw) / 100.0), int(h * (y + rh) / 100.0)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, max(x0 + 1, x1)), min(h, max(y0 + 1, y1))
        return frame[y0:y1, x0:x1], (x0, y0)

    def detect(
        self, frame: np.ndarray, roi: Tuple[float, float, float, float] | None = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Return (detections, stats). Box ternormalisasi 0-100 thd frame penuh."""
        assert self.model is not None, "model not loaded"
        t0 = time.monotonic()
        frame = self.upscale_small(frame)
        fh, fw = frame.shape[:2]
        crop, (ox, oy) = self.crop_roi(frame, roi)
        ch, cw = crop.shape[:2]
        results = self.model.predict(crop, imgsz=config.INFER_IMGSZ, conf=config.CONFIDENCE,
                                     iou=config.IOU, verbose=False)
        dets: List[Dict[str, Any]] = []
        confs: List[float] = []
        r = results[0] if results else None
        if r is not None and r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c, cf in zip(xyxy, cls, conf):
                gx1, gy1 = (x1 + ox) / fw * 100.0, (y1 + oy) / fh * 100.0
                gx2, gy2 = (x2 + ox) / fw * 100.0, (y2 + oy) / fh * 100.0
                dets.append({
                    "class_id": int(c),
                    "label": self.class_names.get(int(c), str(int(c))),
                    "category": self.category_of(int(c)),
                    "confidence": round(float(cf), 3),
                    "box": [round(gx1, 1), round(gy1, 1), round(gx2, 1), round(gy2, 1)],
                })
                confs.append(float(cf))
        stats = {
            "count": len(dets),
            "vehicles": sum(1 for d in dets if d["category"] == "vehicle"),
            "latency_s": round(time.monotonic() - t0, 3),
            "confidence_avg": round(sum(confs) / len(confs), 3) if confs else 0.0,
        }
        return dets, stats
