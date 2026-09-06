#!/usr/bin/env python3
"""Rotasi token kamera: hash baru ke DynamoDB, token tampil SEKALI di stdout.

Jangan commit/print token ke file/report. Simpan di env 600 / sampaikan ke owner.
Contoh: python scripts/rotate_camera_token.py --camera-id CAM_TALUN_NORTH_01
"""
from __future__ import annotations

import argparse
import secrets
import sys

sys.path.insert(0, ".")

import boto3

from app.camera_registry import hash_token
from app import config


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-id", required=True)
    args = ap.parse_args()
    token = secrets.token_urlsafe(32)
    table = boto3.resource("dynamodb", region_name=config.AWS_REGION).Table(config.CAMERAS_TABLE)
    resp = table.get_item(Key={"camera_id": args.camera_id})
    if "Item" not in resp:
        print(f"unknown camera: {args.camera_id}", file=sys.stderr)
        return 1
    table.update_item(
        Key={"camera_id": args.camera_id},
        UpdateExpression="SET device_token_hash = :h",
        ExpressionAttributeValues={":h": hash_token(token)},
    )
    print(f"camera_id: {args.camera_id}")
    print("NEW TOKEN (simpan sekali):")
    print(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
