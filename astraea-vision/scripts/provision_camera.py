#!/usr/bin/env python3
"""Provisioning kamera: buat entri registry + cetak TOKEN SEKALI (PRD §32/CAM-FW-02).

Token plaintext TIDAK disimpan di mana pun — hanya hash SHA-256 di DynamoDB.
Contoh:
  python scripts/provision_camera.py --camera-id CAM_TALUN_NORTH_01 \\
      --intersection-id SIMPANG_TALUN_01 --approach-id north
"""
from __future__ import annotations

import argparse
import secrets
import sys
import time

sys.path.insert(0, ".")

import boto3

from app.camera_registry import hash_token
from app import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", required=True)
    ap.add_argument("--intersection-id", required=True)
    ap.add_argument("--approach-id", required=True)
    ap.add_argument("--firmware", default="")
    args = ap.parse_args()

    token = secrets.token_urlsafe(32)
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    table = boto3.resource("dynamodb", region_name=config.AWS_REGION).Table(config.CAMERAS_TABLE)
    table.put_item(Item={
        "camera_id": args.camera_id,
        "intersection_id": args.intersection_id,
        "approach_id": args.approach_id,
        "device_token_hash": hash_token(token),
        "enabled": True,
        "firmware_version": args.firmware,
        "health_state": "OFFLINE",
        "created_at": now,
    })
    print("camera_id:", args.camera_id)
    print("TOKEN (catat sekali, tidak bisa dilihat lagi):")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
