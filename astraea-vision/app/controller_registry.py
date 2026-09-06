"""Resolusi controller per intersection dari tabel Intersections (cache 5 menit)."""
from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import boto3

from . import config

logger = logging.getLogger("vision.controller")

_cache: Dict[str, Tuple[float, str]] = {}
_table = None


def _table_ref():
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb", region_name=config.AWS_REGION).Table("Intersections")
    return _table


def controller_for(intersection_id: str) -> str:
    now = time.monotonic()
    hit = _cache.get(intersection_id)
    if hit and now - hit[0] < 300:
        return hit[1]
    fallback = config._getenv("VISION_CONTROLLER_ID", "ESP32_TRAFFIC_01")
    try:
        resp = _table_ref().get_item(Key={"intersection_id": intersection_id})
        item = resp.get("Item") or {}
        cid = str(item.get("device_id") or item.get("deviceId") or fallback)
    except Exception as exc:
        logger.warning("controller lookup failed: %s", exc)
        cid = fallback
    _cache[intersection_id] = (now, cid)
    return cid
