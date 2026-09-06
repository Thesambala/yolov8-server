"""Auth kamera: validasi identitas + token vs registry, cache singkat (PRD §32)."""
from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any, Dict, Optional, Tuple

from . import camera_registry, config


_cache: Dict[str, Tuple[float, Optional[Dict[str, Any]]]] = {}


def verify_camera(
    camera_id: str,
    intersection_id: str,
    approach_id: str,
    token: str,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """Return (ok, reason, camera_item). Tidak pernah log token asli."""
    now = time.monotonic()
    hit = _cache.get(camera_id)
    if hit and now - hit[0] < config.AUTH_CACHE_S:
        cam = hit[1]
    else:
        try:
            cam = camera_registry.get_camera(camera_id)
        except Exception as exc:  # DB down -> tolak tegas, catat alasan
            return False, f"registry-unavailable: {type(exc).__name__}", None
        _cache[camera_id] = (now, cam)
    if not cam:
        return False, "unknown-camera", None
    if not cam.get("enabled", False):
        return False, "camera-disabled", None
    if cam.get("intersection_id") != intersection_id or cam.get("approach_id") != approach_id:
        return False, "identity-mismatch", None
    want = str(cam.get("device_token_hash", ""))
    got = hashlib.sha256((token or "").encode("utf-8")).hexdigest()
    if not want or not hmac.compare_digest(want, got):
        return False, "invalid-token", None
    return True, "ok", cam
