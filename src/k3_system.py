import cv2
from ultralytics import YOLO

print("Loading model... harap tunggu")

pose_model = YOLO('yolov8n-pose.pt')
ppe_model  = YOLO('ppe_model.pt')

KELAS_APD_TIDAK_LENGKAP = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']

def cek_jatuh(keypoints):
    if keypoints is None or len(keypoints) < 13:
        return False

    nose_y       = keypoints[0][1]
    bahu_kiri_y  = keypoints[5][1]
    bahu_kanan_y = keypoints[6][1]
    hip_kiri_y   = keypoints[11][1]
    hip_kanan_y  = keypoints[12][1]

    # Semua titik harus terdeteksi
    if nose_y < 5 or hip_kiri_y < 5 or hip_kanan_y < 5:
        return False

    hip_avg      = (hip_kiri_y + hip_kanan_y) / 2
    bahu_avg     = (bahu_kiri_y + bahu_kanan_y) / 2
    tinggi_tubuh = abs(nose_y - hip_avg)
    lebar_bahu   = abs(keypoints[5][0] - keypoints[6][0])

    # Kondisi 1: hidung lebih rendah dari pinggul (jatuh total)
    if nose_y > hip_avg:
        return True

    # Kondisi 2: bahu hampir sejajar atau lebih rendah dari pinggul
    if bahu_avg > hip_avg * 0.85:
        return True

    # Kondisi 3: posisi tubuh mendatar (tidur/rebah)
    if lebar_bahu > 0 and tinggi_tubuh < lebar_bahu * 0.5:
        return True

    return False

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam tidak terdeteksi!")
    exit()

print("Sistem K3 AI berjalan! Tekan Q untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    peringatan = []

    # ── 1. FALL DETECTION ────────────────────────────────────
    pose_hasil = pose_model(frame, verbose=False)

    if pose_hasil[0].keypoints is not None:
        for orang_kps in pose_hasil[0].keypoints.xy:
            kps = orang_kps.cpu().numpy()
            if cek_jatuh(kps):
                peringatan.append("!! PEKERJA JATUH !!")

    # ── 2. PPE DETECTION ─────────────────────────────────────
    ppe_hasil = ppe_model(frame, conf=0.45, verbose=False)

    for box in ppe_hasil[0].boxes:
        cls_id     = int(box.cls[0])
        nama_kelas = ppe_model.names[cls_id]

        if any(k.lower() in nama_kelas.lower() for k in KELAS_APD_TIDAK_LENGKAP):
            peringatan.append(f"APD TIDAK LENGKAP: {nama_kelas}")
            break

    # ── 3. TAMPILAN ──────────────────────────────────────────
    frame_tampil   = ppe_hasil[0].plot()
    pose_annotated = pose_hasil[0].plot()
    frame_tampil   = cv2.addWeighted(frame_tampil, 0.6, pose_annotated, 0.4, 0)

    if peringatan:
        overlay       = frame_tampil.copy()
        tinggi_banner = 45 + len(peringatan) * 35
        cv2.rectangle(overlay, (0, 0), (frame_tampil.shape[1], tinggi_banner),
                      (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame_tampil, 0.3, 0, frame_tampil)

        cv2.putText(frame_tampil, "PERINGATAN!", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for i, p in enumerate(peringatan):
            cv2.putText(frame_tampil, p, (10, 55 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.rectangle(frame_tampil, (0, 0), (frame_tampil.shape[1], 45),
                      (0, 100, 0), -1)
        cv2.putText(frame_tampil, "KONDISI AMAN - Semua APD Terpasang",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    h, w = frame_tampil.shape[:2]
    cv2.putText(frame_tampil, "K3 AI Monitor v1.0", (w - 220, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Sistem K3 AI - Monitor Keselamatan Kerja", frame_tampil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistem dimatikan.")
