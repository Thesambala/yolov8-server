# Aturan Wajib Proyek ASTRAEA (berlaku untuk SEMUA AI & SEMUA sesi)

YOLOv8 inference server: WebSocket :8080 + HTTP :8081 + MQTT bridge ke broker EC2 #1.

1. COMMIT + PUSH SETIAP PERUBAHAN: setiap file yang diubah/ditambah (kode, config, docs)
   WAJIB di-`git commit` dan `git push` ke branch yang sama sebelum sesi/pekerjaan selesai.
   Dilarang menumpuk perubahan tanpa push.
2. SECRET: semua repo kini PRIVATE. Kredensial (MQTT_* systemd, API key) BOLEH
   didokumentasikan di repo selama tetap private. DILARANG menjadikan repo ini public.
   Jangan commit file model (`*.pt`, puluhan MB) — model produksi
   `yolov8s_indotraffic_best.pt` (kelas: Bus, Mobil Penumpang, Pejalan Kaki,
   Sepeda Motor, Truck, Unmotorized) tersimpan di aslamrosul/backup-all.
3. LOKASI DEPLOY: EC2 `yolov8-server` (ap-southeast-2, tanpa key `.pem` di operator).
   Akses via `aws ec2-instance-connect send-ssh-public-key` (user `ubuntu`, berlaku ~60 detik).
   Service systemd: `yolov8-ws` (:8080, `server.py`), `yolov8-http` (:8081, via `uvicorn
   server_detect_http:app` — JANGAN dijalankan via `python` langsung),
   `yolov8-bridge` (publish ke `traffic/CAM_YOLO_01/data`).
   Dependensi: `~/yolo-env` (ultralytics, torch CPU, opencv-headless + `libgl1` sistem).
4. AUTO-PUSH: server ini punya deploy key SSH (`~/.ssh/astraea-yolo`, remote `gh-yolo`).
   Selesai mengubah file di `~/workspace/yolov8-server`: commit lalu `git push`
   langsung (tanpa token). Lalu sync ke EC2 `yolov8-server` via EC2 Instance Connect
   (scp + `systemctl restart yolov8-ws yolov8-http yolov8-bridge`).
