"""Regression freshness: seq claim, frame-age source of truth, DEGRADED/FALLBACK (I)."""
import sys
import time

sys.path.insert(0, ".")

from app.frame_store import FrameStore
from app.metrics import intersection_vision_state

JPEG = b"\xff\xd8\xff\xe0FAKEJPEG"


def test_claim_same_frame_once():
    fs = FrameStore()
    fs.put("C1", JPEG)
    first = fs.claim_frame("C1", max_age_s=60.0)
    assert first is not None and first["seq"] == 1
    assert fs.claim_frame("C1", max_age_s=60.0) is None  # sama -> skip


def test_claim_new_seq_after_new_frame():
    fs = FrameStore()
    fs.put("C1", JPEG)
    fs.claim_frame("C1", max_age_s=60.0)
    fs.put("C1", JPEG)
    second = fs.claim_frame("C1", max_age_s=60.0)
    assert second is not None and second["seq"] == 2


def test_fresh_frame_online_semantics():
    fs = FrameStore()
    fs.put("C1", JPEG)
    age = fs.age_s("C1")
    assert age is not None and age < 15.0  # VISION_FRESH_S default


def test_stale_frame_rejected():
    fs = FrameStore()
    fs.put("C1", JPEG)
    time.sleep(0.15)
    assert fs.claim_frame("C1", max_age_s=0.05) is None  # basi -> jangan infer
    # Snapshot (get) tetap tersedia sebagai last-known:
    assert fs.get("C1") is not None


def test_degraded_two_of_three():
    state, n = intersection_vision_state([True, True, False])
    assert (state, n) == ("DEGRADED", 2)


def test_fallback_all_stale():
    state, n = intersection_vision_state([False, False, False])
    assert (state, n) == ("FALLBACK", 0)


def test_normal_all_fresh():
    state, n = intersection_vision_state([True, True, True])
    assert (state, n) == ("NORMAL", 3)


def test_reconnect_new_seq():
    fs = FrameStore()
    fs.put("C1", JPEG)
    fs.claim_frame("C1", max_age_s=60.0)
    # ...waktu berlalu, kamera kirim lagi:
    fs.put("C1", JPEG)
    item = fs.claim_frame("C1", max_age_s=60.0)
    assert item is not None and item["seq"] == 2
