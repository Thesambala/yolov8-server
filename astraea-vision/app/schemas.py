"""Skema kontrak data (PRD §15/24/27). Validasi ringan, tanpa dependensi berat."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_approach_state(
    approach_id: str,
    camera: Dict[str, Any],
    sensor: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "approach_id": approach_id,
        "camera": camera,
        "sensor": sensor,
    }


def build_recommendation(
    intersection_id: str,
    controller_id: str,
    approaches: Dict[str, Dict[str, Any]],
    schema_version: int = 1,
    valid_for_ms: int = 10000,
) -> Dict[str, Any]:
    return {
        "schema_version": schema_version,
        "intersection_id": intersection_id,
        "controller_id": controller_id,
        "generated_at": utc_now_iso(),
        "valid_for_ms": valid_for_ms,
        "approaches": approaches,
    }


def validate_recommendation(rec: Dict[str, Any], intersection_id: str, controller_id: str) -> List[str]:
    """Kembalikan daftar error (kosong = valid). Dipakai controller & test."""
    errors: List[str] = []
    if rec.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    if rec.get("intersection_id") != intersection_id:
        errors.append("intersection_id mismatch")
    if rec.get("controller_id") != controller_id:
        errors.append("controller_id mismatch")
    try:
        gen = datetime.fromisoformat(str(rec.get("generated_at", "")))
        age_ms = (datetime.now(timezone.utc) - gen).total_seconds() * 1000.0
        if age_ms > float(rec.get("valid_for_ms", 0)):
            errors.append("recommendation expired")
    except (ValueError, TypeError):
        errors.append("generated_at invalid")
    if not isinstance(rec.get("approaches"), dict) or not rec["approaches"]:
        errors.append("approaches missing")
    return errors


def empty_camera_metrics() -> Dict[str, Any]:
    return {
        "active_vehicle_count": 0,
        "flow_count_60s": 0,
        "queue_vehicle_count": 0,
        "stopped_vehicle_count": 0,
        "max_waiting_time_s": 0.0,
        "confidence": 0.0,
        "online": False,
    }
