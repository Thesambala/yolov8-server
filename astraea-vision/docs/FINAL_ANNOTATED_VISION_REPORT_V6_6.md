# FINAL_ANNOTATED_VISION_REPORT_V6_6.md (yolov8-server)

Nilai secret tidak ada di laporan ini.

## 1. HEAD before work

`0c43543` (clean).

## 2. Raw-preview limitation

`/snapshot.jpg` + `/stream.mjpeg` mengembalikan frame mentah FrameStore:
count terlihat, box tidak.

## 3. Single-inference architecture

Satu `track_frame()` per frame → tracks dipakai metrik + fuzzy + anotasi.
Dilarang `detect()` kedua; dibuktikan test + grep diff (tanpa pemanggilan baru).

## 4. Track bbox schema

`bbox: {x1,y1,x2,y2}` ternormalisasi 0..1 full-frame (ox/oy crop + clamp).
Field lama (cx/cy/label/category/confidence) tak berubah.

## 5. Annotation renderer

`app/annotator.py` murni: rectangle + `label pct #id` (+ STOPPED),
garis COUNT LINE opsional, debug teks hanya bila `ANNOTATION_DEBUG=true`
(default false). Tak ada YOLO di modul ini.

## 6. Annotated store

`AnnotatedFrameStore` terpisah: latest-only per kamera
(jpeg/at/wall/bytes/width/height/source_seq/track_count), lock sendiri,
bytes immutable. Raw store tak tersentuh.

## 7. source_seq provenance

`source_seq = item["seq"]` frame yang diinferensi; status publikasikan
`annotated_available/age/source_seq`.

## 8. Server endpoints

`GET /v1/cameras/{id}/annotated.jpg` (auth viewer sama; 404 unknown,
404/503 tanpa frame). Tanpa endpoint MJPEG baru (snapshot cadence cukup).

## 9. Authentication

Sama dengan snapshot/stream. Tanpa token di URL.

## 10. Next proxy

Route web mirror (repo web). Token server-side.

## 11-12. Dashboard/Detail integration

Repo web: prefer annotated, fallback raw jujur, tick 1500ms.

## 13. Refresh cadence

1500ms (3 kamera ≈ 2 req/detik total; ringan).

## 14. Performance measurements

Annotate+encode 5 tracks: avg ~2.2ms (p95 ~1.1ms, cold-start skew),
JPEG ~13KB. Inferensi ~700ms+ (640 CPU). Overhead <0.5%.
Fuzzy/MQTT tak terganggu (anotasi setelah metrik + try/except).

## 15. Tests

`tests/test_annotator.py` 6/6: norm/clamp, offset ROI, kosong,
satu+stopped, encode JPEG, store latest-only+seq, single-inference
(`model.track` tepat 1x, anotasi tanpa panggil model).
Suite penuh S2: 33 PASS. py_compile OK.

## 16. Regression audit

Diff: tracking (bbox aditif), annotator (baru), frame_store (store baru),
main (blok anotasi guarded), stream (endpoint + 3 field aditif),
tests. HUB/fuzzy/MQTT/tracker/waiting/queue tak tersentuh.

## 17. Deployment scope

`app/*.py` ke Server 2 + restart astraea-vision (active, model toy aktif
sesuai keputusan owner, warmup ok, MQTT ok, traceback 0).
Tanpa mosquitto/subscriber/firmware.

## 18. Known limitations

Box butuh frame inference fresh (kamera diam = annotated menua jujur).
Align visual final = owner (pronounced boxes vs mobil mainan).

## 19. Final verdict

Anotasi kanonis PASS (software). Visual hardware = owner.
