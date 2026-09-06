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
