"""Fuzzy adaptive engine: antrean + sensor + tunggu -> rekomendasi hijau (PRD §17).

Hanya merekomendasikan DURASI, tidak pernah warna lampu. Output selalu di-clamp ke
batas konfigurasi per-intersection (§17.4). Matriks aturan = default kalibrasi prototype.
"""
from __future__ import annotations

from typing import Any, Dict


def _tri(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b != a else 1.0
    return (c - x) / (c - b) if c != b else 1.0


def fuzz_queue_ratio(q: float) -> Dict[str, float]:
    q = max(0.0, min(1.0, q))
    low = 1.0 if q <= 0.15 else max(0.0, (0.35 - q) / 0.20)
    return {
        "LOW": low,
        "MEDIUM": _tri(q, 0.2, 0.5, 0.8),
        "HIGH": _tri(q, 0.6, 1.0, 1.01),
    }


def fuzz_waiting(w: float) -> Dict[str, float]:
    w = max(0.0, w)
    return {
        "SHORT": 1.0 if w < 10 else max(0.0, (25 - w) / 15.0),
        "MEDIUM": _tri(w, 10, 30, 60),
        "LONG": 0.0 if w < 40 else min(1.0, (w - 40) / 20.0),
    }


def fuzz_sensor(level: int) -> Dict[str, float]:
    return {
        "LEVEL_0": 1.0 if level <= 0 else 0.0,
        "LEVEL_1": 1.0 if level == 1 else 0.0,
        "LEVEL_2": 1.0 if level >= 2 else 0.0,
    }


def infer_green_factor(queue_ratio: float, waiting_s: float, sensor_level: int,
                       vision_online: bool) -> float:
    """Faktor 0..1 (SHORT_GREEN..LONG_GREEN) dari matriks aturan §17.5."""
    q = fuzz_queue_ratio(queue_ratio)
    w = fuzz_waiting(waiting_s)
    s = fuzz_sensor(sensor_level)
    long_a = max(
        min(q["HIGH"], 1.0),
        min(w["LONG"], 1.0),
        min(q["MEDIUM"], s["LEVEL_2"]),
    )
    med_a = max(
        min(q["MEDIUM"], 1.0),
        min(q["LOW"], w["LONG"]),
    )
    if not vision_online:  # fallback konservatif berbatas (§17.5 aturan terakhir)
        if sensor_level >= 2:
            return 0.65
        if sensor_level == 1:
            return 0.45
        return 0.25
    short_w = 1.0 - max(long_a, med_a)
    # Defuzzifikasi centroid sederhana: SHORT=0.2, MEDIUM=0.55, LONG=0.9
    num = 0.9 * long_a + 0.55 * med_a + 0.2 * short_w
    den = long_a + med_a + short_w
    return num / den if den > 0 else 0.2


def recommend_green(queue_ratio: float, waiting_s: float, sensor_level: int,
                    vision_online: bool, min_green: float, max_green: float) -> Dict[str, Any]:
    factor = infer_green_factor(queue_ratio, waiting_s, sensor_level, vision_online)
    green = min_green + factor * (max_green - min_green)
    green = max(min_green, min(max_green, green))
    return {
        "recommended_green_s": round(green, 1),
        "factor": round(factor, 3),
        "inputs": {
            "queue_ratio": round(queue_ratio, 3),
            "waiting_s": round(waiting_s, 1),
            "sensor_level": int(sensor_level),
            "vision_online": bool(vision_online),
        },
    }
