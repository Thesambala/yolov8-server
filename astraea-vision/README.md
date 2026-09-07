# astraea-vision-service (PRD §29–30)

Unified vision service Server 2: ingest WSS kamera → YOLOv8 + ByteTrack →
metrik per-pendekatan → fuzzy → MQTT (kanonis + legacy). Menggantikan
`server.py`/`server_detect_http.py`/`mqtt_bridge.py` yang terfragmentasi.

Port: WS ingest `8082` (`/v1/camera/ingest`), HTTP API `8081`
(`/health`, `/v1/cameras/...`, `/v1/control/recommendation/...`, `/detect` kompatibel).

Jalankan lokal (butuh model + AWS):
```
pip install -r requirements.txt
python -m app.main
```

Provisioning kamera (token tampil SEKALI):
```
python scripts/provision_camera.py --camera-id CAM_TALUN_NORTH_01 \
  --intersection-id SIMPANG_TALUN_01 --approach-id north
```

Uji: `python -m pytest tests/ -q` (atau runner tanpa pytest, lihat AGENTS.md).
Deploy: `systemd/astraea-vision.service` → `/home/ubuntu/astraea-vision` di Server 2.

## Vision Lab / dashboard browser via vision.astraea.my.id

Frontend (Vision Lab, dashboard, dashboard2) memakai protokol `server.py`
(kirim `{type:"frame",data}` → balas `{type:"detections"}`), BUKAN protokol
kamera `/v1/camera/ingest` (auth token, tanpa balasan deteksi) dan BUKAN format
`/detect` baru (`box:[...]`). URL default frontend:
`wss://vision.astraea.my.id/yolo-ws/ws`.

Deploy di Server 2 (54.253.237.179):

1. Copy service + nginx, aktifkan:
   ```
   sudo cp systemd/yolov8-ws.service /etc/systemd/system/
   sudo cp nginx-vision-site.conf /etc/nginx/sites-enabled/vision
   sudo systemctl daemon-reload
   sudo systemctl enable --now yolov8-ws
   sudo nginx -t && sudo systemctl reload nginx
   ```
   `yolov8-ws` menjalankan `server.py` (:8080) dengan model produksi
   `yolov8s_indotraffic_best.pt` (kelas Indonesia; inferensi imgsz 640 = train size, tanpa upscale VGA).
2. Cek: `systemctl is-active yolov8-ws` dan dari laptop
   `curl -s https://vision.astraea.my.id/health`.
3. Opsional: nginx Server 1 (`astraea.my.id`) masih proxy `/yolo-ws/` ke IP
   lama `3.106.139.0:8080` yang sudah mati — update ke
   `proxy_pass http://54.253.237.179:8080/;` atau hapus blok itu bila frontend
   sudah pindah ke `vision.astraea.my.id`.
4. Frontend memakai `NEXT_PUBLIC_YOLO_WS_URL` bila di-set → setelah ubah,
   `npm run build` + `systemctl restart adaptive-traffic` di Server 1.

Catatan: `POST /detect` mengembalikan 500 saat diuji 2026-09-06 — sebelum
diandalkan, cek `journalctl -u astraea-vision -n 100` di Server 2 dan uji
langsung `curl -F frame=@test.jpg http://127.0.0.1:8081/detect`.
