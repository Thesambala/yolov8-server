# Model Registry — Server 2 (`~/models/` + `/home/ubuntu/*.pt`)

YOLOv8s semua (jangan ganti family). Inferensi produksi imgsz 640.

| File | Kelas | Status |
|---|---|---|
| `/home/ubuntu/yolov8s_indotraffic_best.pt` | 6: Bus, Mobil Penumpang, Pejalan Kaki, Sepeda Motor, Truck, Unmotorized (11.14M) | AKTIF (DETECT_MODEL) |
| `~/models/yolov8s_toy-only_best.pt` | 2: Bus, Mobil Penumpang (11.14M, train 640) | ARSIP siap pakai (maket/toy) |

Ganti model aktif (butuh restart service):

```bash
ssh astraea-yolo "sudo sed -i 's|^Environment=DETECT_MODEL=.*|Environment=DETECT_MODEL=/home/ubuntu/models/yolov8s_toy-only_best.pt|' /etc/systemd/system/astraea-vision.service; sudo systemctl daemon-reload; sudo systemctl restart astraea-vision"
```

Kembalikan: ganti path ke `yolov8s_indotraffic_best.pt`, ulangi perintah.
Kedua file juga ada salinannya di Server 1 `~/models/`.
