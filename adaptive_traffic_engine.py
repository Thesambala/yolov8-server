"""
adaptive_traffic_engine.py
========================================================================================
Master Core Engine: YOLOv8/11 Vision Perception + MKJI SMP/PCU Weighting + Fuzzy Logic Signal Control
Author: Adaptive Traffic Monitoring Team
Version: 2.0.0
========================================================================================
Features:
1. YOLOv8/11 Perception Engine with optimized CCTV inference (imgsz=960, conf=0.25, iou=0.60)
2. Indonesian Traffic BBox Fusion (merge_rider_and_motorcycle) to prevent double counting
3. MKJI 1997 / PKJI 2023 Traffic Engineering Standards for Satuan Mobil Penumpang (SMP / PCU)
4. Mamdani Fuzzy Logic Controller with Center of Gravity (COG) Defuzzification (10s - 60s)
5. Starvation Prevention Mechanism (Weighted Waiting Time)
6. Emergency Vehicle Preemption Override (Ambulance / Fire Truck Priority)
========================================================================================
"""

from __future__ import annotations

import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

# ======================================================================================
# 1. KONSTANTA KELAS, BOBOT MKJI (SMP/PCU), & PARAMETER
# ======================================================================================
# Berdasarkan Manual Kapasitas Jalan Indonesia (MKJI 1997 / PKJI 2023):
# - Sepeda Motor (MC)           = 0.50 SMP
# - Kendaraan Ringan / Mobil    = 1.00 SMP
# - Kendaraan Berat / Bus/Truk  = 2.50 SMP
# - Sepeda / Non-Motorized      = 0.20 SMP
MKJI_SMP_WEIGHTS: Dict[str, float] = {
    "motorcycle": 0.50,
    "bicycle": 0.20,
    "car": 1.00,
    "bus": 2.50,
    "truck": 2.50,
    "ambulance": 99.00,
}

COCO_TO_TRAFFIC_CLASS: Dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

DEFAULT_LANES = ["north", "south", "east", "west"]


# ======================================================================================
# 2. RIDER-MOTORCYCLE BBOX FUSION
# ======================================================================================
def merge_rider_and_motorcycle(
    boxes: np.ndarray,
    confs: np.ndarray,
    cls_ids: np.ndarray,
    iou_thresh: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menggabungkan bounding box 'person' (pengendara) yang menempel atau berada di atas
    'motorcycle' menjadi satu bounding box tunggal 'motorcycle' agar tidak terhitung ganda.
    """
    if len(boxes) == 0:
        return boxes, confs, cls_ids

    moto_indices = [i for i, c in enumerate(cls_ids) if int(c) in [1, 3]]  # bicycle / motorcycle
    person_indices = [i for i, c in enumerate(cls_ids) if int(c) == 0]     # person
    other_indices = [i for i, c in enumerate(cls_ids) if int(c) in [2, 5, 7]]  # car, bus, truck

    if not moto_indices or not person_indices:
        return boxes, confs, cls_ids

    merged_boxes = []
    merged_confs = []
    merged_cls_ids = []
    merged_person_indices = set()

    for m_idx in moto_indices:
        mbox = boxes[m_idx].copy()
        mconf = float(confs[m_idx])
        m_cid = cls_ids[m_idx]

        mx1, my1, mx2, my2 = mbox
        mw = max(1.0, mx2 - mx1)
        mh = max(1.0, my2 - my1)

        for p_idx in person_indices:
            if p_idx in merged_person_indices:
                continue

            px1, py1, px2, py2 = boxes[p_idx]
            pw = max(1.0, px2 - px1)
            ph = max(1.0, py2 - py1)
            p_area = pw * ph

            # Hitung irisan (Intersection)
            ix1, iy1 = max(mx1, px1), max(my1, py1)
            ix2, iy2 = min(mx2, px2), min(my2, py2)
            iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
            inter_area = iw * ih

            overlap = inter_area / p_area if p_area > 0 else 0.0
            pcx = (px1 + px2) / 2.0
            h_align = (mx1 - 0.40 * mw) <= pcx <= (mx2 + 0.40 * mw)
            v_prox = (py2 >= my1 - 0.50 * mh) and (py1 <= my2 + 0.20 * mh)

            if overlap >= iou_thresh or (h_align and v_prox):
                # Perluas box motor untuk mencakup pengendara
                mbox[0] = min(mbox[0], px1)
                mbox[1] = min(mbox[1], py1)
                mbox[2] = max(mbox[2], px2)
                mbox[3] = max(mbox[3], py2)
                mconf = max(mconf, float(confs[p_idx]))
                merged_person_indices.add(p_idx)

        merged_boxes.append(mbox)
        merged_confs.append(mconf)
        merged_cls_ids.append(m_cid)

    # Masukkan sisa pejalan kaki yang bukan pengendara
    for p_idx in person_indices:
        if p_idx not in merged_person_indices:
            merged_boxes.append(boxes[p_idx])
            merged_confs.append(confs[p_idx])
            merged_cls_ids.append(cls_ids[p_idx])

    # Masukkan kendaraan lain
    for o_idx in other_indices:
        merged_boxes.append(boxes[o_idx])
        merged_confs.append(confs[o_idx])
        merged_cls_ids.append(cls_ids[o_idx])

    return (
        np.array(merged_boxes, dtype=np.float32),
        np.array(merged_confs, dtype=np.float32),
        np.array(merged_cls_ids, dtype=np.int32),
    )


# ======================================================================================
# 3. FUZZY LOGIC MAMDANI CONTROLLER
# ======================================================================================
class FuzzyMamdaniSignalController:
    """
    Fuzzy Logic Controller untuk Pengaturan Waktu Sinyal Lampu Lalu Lintas Adaptif.
    - Input 1: Beban Kendaraan dalam Satuan Mobil Penumpang (SMP / PCU)
    - Input 2: Waktu Tunggu Kendaraan di Lampu Merah (Detik)
    - Output : Alokasi Durasi Lampu Hijau (10 - 60 Detik)
    """

    def __init__(self, min_green: int = 10, max_green: int = 60):
        self.min_green = min_green
        self.max_green = max_green

    # --- 1. Fuzzifikasi Input: Beban Volume (SMP) ---
    def _fuzzy_volume(self, smp: float) -> Dict[str, float]:
        # Lengang (0 - 8 SMP)
        if smp <= 4:
            mu_lengang = 1.0
        elif 4 < smp < 8:
            mu_lengang = (8.0 - smp) / 4.0
        else:
            mu_lengang = 0.0

        # Sedang (6 - 18 SMP)
        if 6 <= smp <= 12:
            mu_sedang = (smp - 6.0) / 6.0
        elif 12 < smp <= 18:
            mu_sedang = (18.0 - smp) / 6.0
        else:
            mu_sedang = 0.0

        # Padat (15 - 30 SMP)
        if 15 <= smp <= 22:
            mu_padat = (smp - 15.0) / 7.0
        elif 22 < smp <= 30:
            mu_padat = (30.0 - smp) / 8.0
        else:
            mu_padat = 0.0

        # Sangat Padat (>= 25 SMP)
        if smp <= 25:
            mu_sangat_padat = 0.0
        elif 25 < smp < 35:
            mu_sangat_padat = (smp - 25.0) / 10.0
        else:
            mu_sangat_padat = 1.0

        return {
            "lengang": float(np.clip(mu_lengang, 0.0, 1.0)),
            "sedang": float(np.clip(mu_sedang, 0.0, 1.0)),
            "padat": float(np.clip(mu_padat, 0.0, 1.0)),
            "sangat_padat": float(np.clip(mu_sangat_padat, 0.0, 1.0)),
        }

    # --- 2. Fuzzifikasi Input: Waktu Tunggu (Detik) ---
    def _fuzzy_waiting(self, wait_sec: float) -> Dict[str, float]:
        # Sebentar (0 - 30 detik)
        if wait_sec <= 20:
            mu_sebentar = 1.0
        elif 20 < wait_sec < 35:
            mu_sebentar = (35.0 - wait_sec) / 15.0
        else:
            mu_sebentar = 0.0

        # Sedang (25 - 65 detik)
        if 25 <= wait_sec <= 45:
            mu_sedang = (wait_sec - 25.0) / 20.0
        elif 45 < wait_sec <= 65:
            mu_sedang = (65.0 - wait_sec) / 20.0
        else:
            mu_sedang = 0.0

        # Lama (>= 50 detik)
        if wait_sec <= 50:
            mu_lama = 0.0
        elif 50 < wait_sec < 80:
            mu_lama = (wait_sec - 50.0) / 30.0
        else:
            mu_lama = 1.0

        return {
            "sebentar": float(np.clip(mu_sebentar, 0.0, 1.0)),
            "sedang": float(np.clip(mu_sedang, 0.0, 1.0)),
            "lama": float(np.clip(mu_lama, 0.0, 1.0)),
        }

    # --- 3. Evaluasi Aturan & Defuzzifikasi ---
    def compute_green_duration(
        self,
        smp: float,
        wait_sec: float,
        is_emergency: bool = False,
    ) -> int:
        """
        Menghitung durasi lampu hijau optimal dalam detik menggunakan Centroid / Center of Area.
        """
        if is_emergency:
            return 30  # Alokasi darurat otomatis (Preemption)

        vol = self._fuzzy_volume(smp)
        wt = self._fuzzy_waiting(wait_sec)

        # Output Singletons (Durasi Hijau)
        output_levels = {
            "sangat_singkat": 12.0,
            "singkat": 20.0,
            "sedang": 32.0,
            "lama": 45.0,
            "maksimum": 60.0,
        }

        # Basis Aturan Mamdani (4 x 3 = 12 Aturan)
        rules = [
            (min(vol["lengang"], wt["sebentar"]), output_levels["sangat_singkat"]),
            (min(vol["lengang"], wt["sedang"]), output_levels["singkat"]),
            (min(vol["lengang"], wt["lama"]), output_levels["sedang"]),

            (min(vol["sedang"], wt["sebentar"]), output_levels["singkat"]),
            (min(vol["sedang"], wt["sedang"]), output_levels["sedang"]),
            (min(vol["sedang"], wt["lama"]), output_levels["lama"]),

            (min(vol["padat"], wt["sebentar"]), output_levels["sedang"]),
            (min(vol["padat"], wt["sedang"]), output_levels["lama"]),
            (min(vol["padat"], wt["lama"]), output_levels["maksimum"]),

            (min(vol["sangat_padat"], wt["sebentar"]), output_levels["lama"]),
            (min(vol["sangat_padat"], wt["sedang"]), output_levels["maksimum"]),
            (min(vol["sangat_padat"], wt["lama"]), output_levels["maksimum"]),
        ]

        numerator = sum(weight * val for weight, val in rules)
        denominator = sum(weight for weight, val in rules)

        if denominator == 0.0:
            return self.min_green

        crisp_green = numerator / denominator
        return int(np.clip(round(crisp_green), self.min_green, self.max_green))


# ======================================================================================
# 4. KONTROLER INTERSEKSI MULTI-LAJUR (Thread-Safe State Machine)
# ======================================================================================
@dataclass
class LaneState:
    name: str
    vehicle_counts: Dict[str, int] = field(default_factory=lambda: {
        "motorcycle": 0, "car": 0, "bus": 0, "truck": 0, "bicycle": 0
    })
    total_vehicles: int = 0
    smp_score: float = 0.0
    signal: str = "RED"  # RED | YELLOW | GREEN
    waiting_time_sec: float = 0.0
    emergency_active: bool = False
    last_updated: float = field(default_factory=time.time)


class MultiLaneTrafficController:
    """
    Manajer Persimpangan Cerdas Multi-Lajur terpusat dengan Logika Fuzzy & Emergency Override.
    """

    def __init__(
        self,
        lane_names: Optional[List[str]] = None,
        min_green: int = 10,
        max_green: int = 60,
        yellow_duration: int = 3,
    ):
        self.lane_names = lane_names or DEFAULT_LANES
        self.fuzzy_engine = FuzzyMamdaniSignalController(min_green=min_green, max_green=max_green)
        self.yellow_duration = yellow_duration
        self.lock = threading.Lock()

        self.lanes: Dict[str, LaneState] = {
            name: LaneState(name=name) for name in self.lane_names
        }

        self.active_green_lane: str = self.lane_names[0]
        self.lanes[self.active_green_lane].signal = "GREEN"
        self.allocated_green_sec: int = min_green
        self.phase_start_time: float = time.time()
        self.is_yellow: bool = False
        self.emergency_override: bool = False
        self.emergency_lane: Optional[str] = None

    def update_lane_detection(
        self,
        lane_name: str,
        counts: Dict[str, int],
        emergency: bool = False,
    ) -> None:
        """Menerima hasil deteksi terbaru dari Vision Server."""
        with self.lock:
            if lane_name not in self.lanes:
                self.lanes[lane_name] = LaneState(name=lane_name)

            ls = self.lanes[lane_name]
            ls.vehicle_counts = counts
            ls.total_vehicles = sum(counts.values())
            ls.emergency_active = emergency
            ls.last_updated = time.time()

            # Hitung Satuan Mobil Penumpang (MKJI SMP / PCU)
            smp = 0.0
            for v_type, count in counts.items():
                w = MKJI_SMP_WEIGHTS.get(v_type.lower(), 1.0)
                smp += count * w
            ls.smp_score = round(smp, 2)

    def tick(self) -> Dict[str, Any]:
        """
        Memajukan siklus state machine pengatur lampu (Dipanggil secara periodik, misal tiap detik).
        """
        with self.lock:
            now = time.time()
            elapsed = now - self.phase_start_time

            # Update waktu tunggu untuk lajur merah
            for name, ls in self.lanes.items():
                if ls.signal == "RED":
                    ls.waiting_time_sec = max(0.0, ls.waiting_time_sec + 1.0)
                else:
                    ls.waiting_time_sec = 0.0

            # 1. Evaluasi Emergency Override
            emergency_candidates = [n for n, l in self.lanes.items() if l.emergency_active]
            if emergency_candidates and not self.emergency_override:
                chosen = emergency_candidates[0]
                if self.active_green_lane != chosen:
                    self._switch_lane(chosen, green_duration=30, is_emergency=True)
                    return self.get_status()

            # 2. Handle Fase Kuning
            if self.is_yellow:
                if elapsed >= self.yellow_duration:
                    self.is_yellow = False
                    self._select_and_activate_next_green()
                return self.get_status()

            # 3. Handle Waktu Hijau Habis
            if elapsed >= self.allocated_green_sec:
                self.is_yellow = True
                self.lanes[self.active_green_lane].signal = "YELLOW"
                self.phase_start_time = now

            return self.get_status()

    def _select_and_activate_next_green(self) -> None:
        """Memilih lajur berikutnya berdasarkan kombinasi SMP & waktu tunggu (Anti-Starvation)."""
        candidates = [n for n in self.lane_names if n != self.active_green_lane]
        if not candidates:
            candidates = self.lane_names

        # Priority Score = SMP + (Waktu Tunggu * 0.20)
        best_lane = max(
            candidates,
            key=lambda n: self.lanes[n].smp_score + (self.lanes[n].waiting_time_sec * 0.20),
        )

        target_lane = self.lanes[best_lane]
        green_duration = self.fuzzy_engine.compute_green_duration(
            smp=target_lane.smp_score,
            wait_sec=target_lane.waiting_time_sec,
            is_emergency=target_lane.emergency_active,
        )

        self._switch_lane(best_lane, green_duration=green_duration, is_emergency=target_lane.emergency_active)

    def _switch_lane(self, lane_name: str, green_duration: int, is_emergency: bool = False) -> None:
        self.active_green_lane = lane_name
        self.allocated_green_sec = green_duration
        self.phase_start_time = time.time()
        self.emergency_override = is_emergency
        self.emergency_lane = lane_name if is_emergency else None

        for name, ls in self.lanes.items():
            if name == lane_name:
                ls.signal = "GREEN"
                ls.waiting_time_sec = 0.0
            else:
                ls.signal = "RED"

    def get_status(self) -> Dict[str, Any]:
        """Format data lengkap untuk API endpoint /status dan Web Dashboard."""
        now = time.time()
        elapsed = now - self.phase_start_time
        remaining = max(0, int(self.allocated_green_sec - elapsed)) if not self.is_yellow else max(0, int(self.yellow_duration - elapsed))

        lanes_data = {}
        for name, ls in self.lanes.items():
            lanes_data[name] = {
                "signal": ls.signal,
                "total_vehicles": ls.total_vehicles,
                "smp_load": ls.smp_score,
                "vehicle_counts": ls.vehicle_counts,
                "waiting_seconds": int(ls.waiting_time_sec),
                "emergency": ls.emergency_active,
            }

        return {
            "active_green_lane": self.active_green_lane,
            "phase": "YELLOW" if self.is_yellow else ("EMERGENCY" if self.emergency_override else "GREEN"),
            "allocated_green_seconds": self.allocated_green_sec,
            "time_remaining_seconds": remaining,
            "emergency_active": self.emergency_override,
            "emergency_lane": self.emergency_lane,
            "lanes": lanes_data,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }


# ======================================================================================
# 5. VISION DETECTOR WRAPPER (YOLOv8 / YOLO11)
# ======================================================================================
class VisionTrafficDetector:
    """
    Wrapper inferensi YOLOv8/11 dengan parameter teroptimasi untuk CCTV/ESP32-CAM.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.60,
        imgsz: int = 960,
    ):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.imgsz = imgsz
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            from ultralytics import YOLO
            if os.path.exists(self.model_path):
                self._model = YOLO(self.model_path)
            else:
                # Fallback to yolov8n.pt
                self._model = YOLO("yolov8n.pt")
        except Exception as exc:
            print(f"[VisionTrafficDetector] Model loading warning: {exc}")

    def detect_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Menjalankan deteksi pada frame gambar, menerapkan Rider-Motorcycle fusion,
        dan menghitung total kendaraan serta bobot SMP.
        """
        if self._model is None:
            return {
                "boxes": [],
                "confs": [],
                "classes": [],
                "counts": {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0, "bicycle": 0},
                "smp_load": 0.0,
                "emergency": False,
            }

        # Jalankan inferensi YOLO
        results = self._model.predict(
            source=frame,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            imgsz=self.imgsz,
            verbose=False,
        )

        raw_boxes = []
        raw_confs = []
        raw_cls_ids = []

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                if cls_id in COCO_TO_TRAFFIC_CLASS:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    raw_boxes.append([x1, y1, x2, y2])
                    raw_confs.append(float(box.conf[0].item()))
                    raw_cls_ids.append(cls_id)

        # Terapkan Rider-Motorcycle Fusion
        f_boxes, f_confs, f_cls = merge_rider_and_motorcycle(
            np.array(raw_boxes, dtype=np.float32),
            np.array(raw_confs, dtype=np.float32),
            np.array(raw_cls_ids, dtype=np.int32),
        )

        counts = {"motorcycle": 0, "car": 0, "bus": 0, "truck": 0, "bicycle": 0}
        det_classes = []
        emergency = False

        for i in range(len(f_cls)):
            cid = int(f_cls[i])
            cname = COCO_TO_TRAFFIC_CLASS.get(cid, "car")
            if cname == "person":
                continue  # Sisa pejalan kaki murni
            if cname in counts:
                counts[cname] += 1
            det_classes.append(cname)

            # Heuristik deteksi ambulans/truk darurat (Wide aspect ratio pada truck/large vehicle)
            x1, y1, x2, y2 = f_boxes[i]
            w, h = x2 - x1, y2 - y1
            if cname == "truck" and w * h > 25000 and (w / max(1.0, h) > 1.8):
                emergency = True

        # Hitung beban SMP
        smp = sum(counts[v] * MKJI_SMP_WEIGHTS.get(v, 1.0) for v in counts)

        return {
            "boxes": f_boxes.tolist() if len(f_boxes) > 0 else [],
            "confs": f_confs.tolist() if len(f_confs) > 0 else [],
            "classes": det_classes,
            "counts": counts,
            "smp_load": round(smp, 2),
            "emergency": emergency,
        }
