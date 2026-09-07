"""Registry kamera di DynamoDB. Simpan HANYA hash token (PRD §32)."""
from __future__ import annotations

import hashlib
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional

import boto3

from . import config


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


_table = None


def table():
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb", region_name=config.AWS_REGION).Table(config.CAMERAS_TABLE)
    return _table


def get_camera(camera_id: str) -> Optional[Dict[str, Any]]:
    resp = table().get_item(Key={"camera_id": camera_id})
    return resp.get("Item")


def list_cameras(intersection_id: Optional[str] = None) -> List[Dict[str, Any]]:
    if intersection_id:
        resp = table().scan(
            FilterExpression="intersection_id = :i",
            ExpressionAttributeValues={":i": intersection_id},
        )
    else:
        resp = table().scan()
    return resp.get("Items", [])


def touch_seen(camera_id: str, fps_ingest: float = 0.0, health_state: str = "ONLINE") -> None:
    table().update_item(
        Key={"camera_id": camera_id},
        UpdateExpression="SET last_seen = :now, fps_ingest = :fps, health_state = :h",
        ExpressionAttributeValues={
            ":now": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            ":fps": Decimal(str(fps_ingest)),
            ":h": health_state,
        },
    )
