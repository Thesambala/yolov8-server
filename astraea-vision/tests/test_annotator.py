"""Tests anotasi: bbox, renderer murni, store, single-inference (STEP 18)."""
import sys

import numpy as np

sys.path.insert(0, ".")

from app.annotator import annotate_frame, encode_jpeg
from app.frame_store import AnnotatedFrameStore


class T:
    """Mimik tensor torch untuk .cpu().numpy().astype()."""

    def __init__(self, arr):
        self._a = np.array(arr)

    def cpu(self):
        return self

    def numpy(self):
        return self._a

    def astype(self, t):
        return self._a.astype(t)


class FakeBoxes:
    def __init__(self, dets):
        self._d = dets
        self.xyxy = T([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in dets] or np.zeros((0, 4)))
        self.cls = T([d.get("cls", 0) for d in dets])
        self.conf = T([d.get("conf", 0.9) for d in dets])
        self.id = T([d.get("tid", -1) for d in dets]) if dets else None

    def __len__(self):
        return len(self._d)


class FakeResult:
    def __init__(self, dets):
        self.boxes = FakeBoxes(dets)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def track(self, *a, **k):
        self.calls += 1
        # 1 box crop (0,0)-(100,100) dalam crop 100x100
        return [FakeResult([{"x1": 10, "y1": 20, "x2": 40, "y2": 60,
                             "cls": 0, "conf": 0.9, "tid": 7}])]


class FakeEngine:
    def __init__(self, model):
        self.model = model
        self.class_names = {0: "car"}

    def upscale_small(self, frame):
        return frame

    def crop_roi(self, frame, roi):
        if not roi:
            return frame, (0, 0)
        h, w = frame.shape[:2]
        x, y, rw, rh = roi
        x0, y0 = int(w * x / 100.0), int(h * y / 100.0)
        return frame[y0:y0 + int(h * rh / 100.0), x0:x0 + int(w * rw / 100.0)], (x0, y0)

    def category_of(self, c):
        return "vehicle"


def frame100():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_bbox_normalized_and_clamped():
    from app.tracking import TrackerHub

    hub = TrackerHub(engine_factory=lambda: FakeEngine(FakeModel()))
    tracks, _ = hub.track_frame("C1", frame100())
    assert len(tracks) == 1
    b = tracks[0]["bbox"]
    assert b == {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6}
    assert all(0.0 <= v <= 1.0 for v in b.values())


def test_bbox_roi_offset():
    from app.tracking import TrackerHub

    hub = TrackerHub(engine_factory=lambda: FakeEngine(FakeModel()))
    # ROI x=10,y=10,w=50,h=50 pada 100x100 -> ox=10, oy=10
    tracks, _ = hub.track_frame("C1", frame100(), roi=(10.0, 10.0, 50.0, 50.0))
    b = tracks[0]["bbox"]
    # (10+10)/100=0.2, (20+10)/100=0.3, (40+10)/100=0.5, (60+10)/100=0.7
    assert b == {"x1": 0.2, "y1": 0.3, "x2": 0.5, "y2": 0.7}


def test_annotator_empty_tracks():
    img = frame100()
    out = annotate_frame(img, [], set())
    assert out.shape == img.shape
    assert (out == img).all()  # tanpa track, frame identik (kecuali mungkin garis tak ada)


def test_annotator_one_track_and_stopped():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    tr = [{"track_id": 7, "class_id": 0, "label": "car", "category": "vehicle",
           "confidence": 0.87, "cx": 25.0, "cy": 40.0,
           "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.4, "y2": 0.6}}]
    out = annotate_frame(img, tr, {7})
    assert out.shape == img.shape
    assert (out != img).any()  # ada gambar
    enc = encode_jpeg(out)
    assert enc is not None and enc[:2] == b"\xff\xd8"
    # Tanpa stopped -> tetap gambar, tanpa label STOPPED (tak crash)
    out2 = annotate_frame(img, tr, set())
    assert (out2 != img).any()


def test_annotated_store_latest_only_and_seq():
    st = AnnotatedFrameStore()
    st.put("C1", b"a", source_seq=1)
    st.put("C1", b"bb", source_seq=2)
    got = st.get("C1")
    assert got is not None and got["jpeg"] == b"bb" and got["source_seq"] == 2
    assert st.age_s("C1") is not None and st.age_s("C1") >= 0
    assert st.get("NOPE") is None
    assert st.age_s("NOPE") is None


def test_single_inference_regression():
    # Satu frame -> model.track dipanggil TEPAT sekali (tanpa YOLO kedua).
    from app.tracking import TrackerHub

    model = FakeModel()
    hub = TrackerHub(engine_factory=lambda: FakeEngine(model))
    tracks, summary = hub.track_frame("C9", frame100())
    assert model.calls == 1
    assert len(tracks) == 1
    # Anotasi memakai hasil yang sama tanpa memanggil model lagi.
    out = annotate_frame(frame100(), tracks, set(summary.get("stopped_ids", [])))
    assert model.calls == 1
    assert out is not None
