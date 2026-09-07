"""Regresi tracking: isolasi multi-kamera, hysteresis garis, waiting aktual (P2.1-3).

Tanpa torch/model: engine palsu + frame numpy kecil. Skip otomatis bila numpy tak ada.
"""
import sys

import numpy as np

sys.path.insert(0, ".")

from app.tracking import CameraTracker, TrackerHub


class FakeBoxes:
    def __init__(self, dets):
        # dets: [(tid, cx, cy)] ternormalisasi 0-100 pada frame 100x100
        self._d = dets
        import numpy as _np
        n = len(dets)
        self.xyxy = _np.array([[d[1] - 1, d[2] - 1, d[1] + 1, d[2] + 1] for d in dets] or _np.zeros((0, 4)))
        self.cls = _np.array([0] * n)
        self.conf = _np.array([0.9] * n)
        self.id = _np.array([d[0] for d in dets]) if n else None

    def __len__(self):
        return len(self._d)


class FakeResult:
    def __init__(self, dets):
        self.boxes = FakeBoxes(dets)


class FakeModel:
    """Script per panggilan: {camera_marker: [[dets_call1], [dets_call2], ...]}."""

    def __init__(self, script):
        self.script = script
        self.calls = []

    def track(self, *a, **k):
        self.calls.append(1)
        key = getattr(self, "_key", None)
        seq = self.script.get(key, [[]])
        idx = len(self.calls) - 1
        dets = seq[min(idx, len(seq) - 1)]
        return [FakeResult(dets)]


class FakeEngine:
    def __init__(self, model):
        self.model = model
        self.class_names = {0: "car"}

    def upscale_small(self, frame):
        return frame

    def crop_roi(self, frame, roi):
        return frame, (0, 0)

    def category_of(self, c):
        return "vehicle"


def make_hub(script):
    models = {}

    def factory():
        m = FakeModel(script)
        models[len(models)] = m
        return FakeEngine(m)

    hub = TrackerHub(engine_factory=factory)
    return hub, models


def frame100():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_engine_per_camera_isolation():
    hub, models = make_hub({})
    hub.track_frame("CAM_A", frame100())
    hub.track_frame("CAM_B", frame100())
    hub.track_frame("CAM_A", frame100())
    assert len(models) == 2  # tepat 1 engine per kamera, reuse sesudahnya


def test_interleaved_state_no_contamination():
    # A melihat track 7 di atas garis; B melihat track 9 di bawah garis.
    seqA = [[(7, 50, 30)], [(7, 50, 30)], [(7, 50, 30)]]
    seqB = [[(9, 50, 70)], [(9, 50, 70)], [(9, 50, 70)]]
    hubs = {}

    def factory_for(key):
        m = FakeModel({"x": seqA if key == "A" else seqB})
        return FakeEngine(m)

    hubA = TrackerHub(engine_factory=lambda: factory_for("A"))
    hubB = TrackerHub(engine_factory=lambda: factory_for("B"))
    for _ in range(3):
        hubA.track_frame("CAM_A", frame100())
        hubB.track_frame("CAM_B", frame100())
    sA = hubA.for_camera("CAM_A").update([{"track_id": 7, "cx": 50, "cy": 30}], 100.0)
    sB = hubB.for_camera("CAM_B").update([{"track_id": 9, "cx": 50, "cy": 70}], 100.0)
    assert sA["tracked"] == 1 and sB["tracked"] == 1
    # Sisi: A di atas (tidak cross), B di bawah
    assert hubA.for_camera("CAM_A").side.get(7) == -1
    assert hubB.for_camera("CAM_B").side.get(9) == 1
    assert sA["flow_total"] == 0 and sB["flow_total"] == 0


def test_crossing_hysteresis_no_jitter_count():
    tr = CameraTracker(line_y=50.0)
    t = 0.0
    # Jitter di sekitar garis dalam band: 49,51,49,52,48 -> TIDAK boleh count.
    for cy in (49, 51, 49, 52, 48, 51, 49):
        t += 0.5
        tr.update([{"track_id": 7, "cx": 50, "cy": cy}], t)
    assert tr.flow_total == 0
    # Transisi penuh atas->bawah: 40 -> 60 (lewati band) = 1, arah down.
    t += 0.5
    tr.update([{"track_id": 7, "cx": 50, "cy": 40}], t)
    t += 0.5
    s = tr.update([{"track_id": 7, "cx": 50, "cy": 60}], t)
    assert s["flow_total"] == 1
    assert s["flow_by_direction"]["down"] == 1
    # Kembali bawah->atas penuh = 1 lagi arah up (bukan jitter liar).
    t += 0.5
    s = tr.update([{"track_id": 7, "cx": 50, "cy": 40}], t)
    assert s["flow_total"] == 2
    assert s["flow_by_direction"]["up"] == 1


def test_waiting_time_from_tracks():
    # Monotonik: 100 muncul, 101+104 diam, 105 bergerak, 106 hilang.
    tr = CameraTracker(line_y=50.0)
    tr.update([{"track_id": 3, "cx": 10, "cy": 10}], 100.0)  # muncul
    tr.update([{"track_id": 3, "cx": 10, "cy": 10}], 101.0)  # mulai diam
    s = tr.update([{"track_id": 3, "cx": 10, "cy": 10}], 104.0)  # diam 3 dtk
    assert 3 in s["stopped_ids"]
    assert s["max_waiting_s"] >= 2.5
    # Bergerak signifikan -> reset stopped.
    s = tr.update([{"track_id": 3, "cx": 10, "cy": 40}], 105.0)
    assert 3 not in s["stopped_ids"]
    assert s["max_waiting_s"] == 0.0
    # Track hilang -> state dibersihkan.
    s = tr.update([], 106.0)
    assert s["tracked"] == 0


def _trk(cx, cy, tid=1, cat="vehicle"):
    return {"track_id": tid, "cx": cx, "cy": cy, "category": cat}


def test_person_still_does_not_raise_vehicle_waiting():
    # Person diam 10 dtk + vehicle BERGERAK di t=115 -> vehicle_max = 0.
    # (Satu update() = satu frame berisi SEMUA track terlihat.)
    tr = CameraTracker(line_y=50.0)
    P = lambda x, y: {"track_id": 9, "cx": x, "cy": y, "category": "person"}
    V = lambda x, y: {"track_id": 1, "cx": x, "cy": y, "category": "vehicle"}
    tr.update([P(10, 10), V(60, 60)], 100.0)
    tr.update([P(10, 10), V(60, 60)], 105.0)  # keduanya still@105
    s = tr.update([P(10, 10), V(60, 90)], 115.0)  # P wait=10, V bergerak
    assert s["vehicle_max_waiting_s"] == 0.0  # TEST A (kode lama: 10)
    assert s["vehicle_stopped_ids"] == []


def test_vehicle_wait_excludes_person():
    # Person still@105 (wait 8 di t=113), vehicle still@110 (wait 3).
    # Metric vehicle HARUS 3, bukan 8 (bukti person tidak mencemari).
    tr = CameraTracker(line_y=50.0)
    P = lambda x, y: {"track_id": 9, "cx": x, "cy": y, "category": "person"}
    V = lambda x, y: {"track_id": 1, "cx": x, "cy": y, "category": "vehicle"}
    tr.update([P(10, 10), V(20, 20)], 100.0)
    tr.update([P(10, 10), V(20, 20)], 105.0)  # still@105 keduanya
    tr.update([P(10, 10), V(20, 50)], 108.0)  # V bergerak vertikal (reset)
    tr.update([P(10, 10), V(20, 50)], 110.0)  # V still@110
    s = tr.update([P(10, 10), V(20, 50)], 113.0)
    assert 2.5 <= s["vehicle_max_waiting_s"] <= 3.5  # TEST B (bukan 8)
    assert s["vehicle_stopped_ids"] == [1]


def test_two_vehicles_max():
    tr = CameraTracker(line_y=50.0)
    tr.update([_trk(10, 10, 1), _trk(30, 30, 2)], 100.0)
    tr.update([_trk(10, 10, 1), _trk(30, 30, 2)], 103.0)  # still@103
    s = tr.update([_trk(10, 10, 1), _trk(30, 30, 2)], 108.0)  # wait 5
    assert s["vehicle_max_waiting_s"] >= 4.5  # TEST C (~5)
    assert sorted(s["vehicle_stopped_ids"]) == [1, 2]


def test_below_threshold_zero():
    tr = CameraTracker(line_y=50.0)
    tr.update([_trk(10, 10, 1)], 100.0)
    s = tr.update([_trk(10, 10, 1)], 101.0)  # diam 1 dtk < 2
    assert 1 not in s["stopped_ids"]
    assert s["max_waiting_s"] == 0.0  # TEST D
    assert s["vehicle_max_waiting_s"] == 0.0


def test_move_resets_and_cleanup():
    tr = CameraTracker(line_y=50.0)
    tr.update([_trk(10, 10, 1)], 100.0)
    tr.update([_trk(10, 10, 1)], 104.0)
    # NOTE: tid sama tapi posisi pindah jauh -> speed tinggi -> reset
    s = tr.update([{"track_id": 1, "cx": 10, "cy": 80}], 105.0)
    assert 1 not in s["stopped_ids"]  # TEST E
    s = tr.update([], 106.0)
    assert s["tracked"] == 0  # TEST F
