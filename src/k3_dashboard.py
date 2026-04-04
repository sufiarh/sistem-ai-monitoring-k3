import cv2
import time
import threading
import subprocess
import numpy as np
from ultralytics import YOLO

# ── Konfigurasi ────────────────────────────────────────────
JUDUL_SISTEM  = "Sistem Monitoring K3 Berbasis AI untuk Deteksi Real-Time APD dan Kecelakaan Kerja"
NAMA_LOKASI   = "Area Produksi - Gedung A"
KAMERA_PINTU  = 0   # ganti RTSP kalau pakai CCTV WiFi
KAMERA_RUANGAN = "http://192.168.43.67:8080/video"   # ganti RTSP kalau pakai CCTV WiFi kedua

# Resolusi tiap feed kamera di dashboard
CAM_W = 640
CAM_H = 400

# Warna (BGR)
MERAH   = (0,   0,   200)
HIJAU   = (0,   160, 0  )
KUNING  = (0,   200, 255)
PUTIH   = (255, 255, 255)
ABU     = (60,  60,  60 )
ABU2    = (30,  30,  30 )
ABU3    = (20,  20,  20 )
BIRU    = (200, 100, 0  )

# ── Load Model ─────────────────────────────────────────────
print("Loading model... harap tunggu")
ppe_model  = YOLO('ppe_model.pt')
pose_model = YOLO('yolov8n-pose.pt')
print("Model siap!")

KELAS_TIDAK_LENGKAP = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']
KELAS_LENGKAP       = ['Hardhat', 'Safety Vest', 'Mask']

# ── State Global ───────────────────────────────────────────
total_pelanggaran = 0
waktu_jatuh       = None
waktu_alarm_apd   = 0
waktu_alarm_fall  = 0
COOLDOWN          = 6
DURASI_ALERT      = 7
waktu_mulai       = time.time()

# ── Alarm ──────────────────────────────────────────────────
def bunyikan_alarm(pesan):
    def _run():
        subprocess.run(['espeak-ng', '-v', 'id', '-s', '130', pesan],
                       capture_output=True)
    threading.Thread(target=_run, daemon=True).start()

# ── Fall Detection ─────────────────────────────────────────
def cek_jatuh(keypoints):
    if keypoints is None or len(keypoints) < 13:
        return False
    nose_y       = keypoints[0][1]
    bahu_kiri_y  = keypoints[5][1]
    bahu_kanan_y = keypoints[6][1]
    hip_kiri_y   = keypoints[11][1]
    hip_kanan_y  = keypoints[12][1]
    if nose_y < 5 or hip_kiri_y < 5 or hip_kanan_y < 5:
        return False
    hip_avg      = (hip_kiri_y + hip_kanan_y) / 2
    bahu_avg     = (bahu_kiri_y + bahu_kanan_y) / 2
    tinggi_tubuh = abs(nose_y - hip_avg)
    lebar_bahu   = abs(keypoints[5][0] - keypoints[6][0])
    if nose_y > hip_avg:
        return True
    if bahu_avg > hip_avg * 0.85:
        return True
    if lebar_bahu > 0 and tinggi_tubuh < lebar_bahu * 0.5:
        return True
    return False

# ── Helper: Teks di tengah ─────────────────────────────────
def teks_tengah(canvas, teks, y, font_scale, warna, tebal=1):
    (tw, th), _ = cv2.getTextSize(teks, cv2.FONT_HERSHEY_SIMPLEX, font_scale, tebal)
    x = (canvas.shape[1] - tw) // 2
    cv2.putText(canvas, teks, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, warna, tebal)

# ── Helper: Buat panel kamera ──────────────────────────────
def buat_panel_kamera(frame, label_cam, label_area, status_teks, status_warna, info_list):
    # Resize frame ke ukuran panel
    panel = cv2.resize(frame, (CAM_W, CAM_H))

    # Header kamera
    cv2.rectangle(panel, (0, 0), (CAM_W, 38), ABU, -1)
    cv2.putText(panel, label_cam,  (10, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, KUNING, 1)
    cv2.putText(panel, label_area, (10, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, PUTIH, 1)

    # Waktu di pojok kanan header
    jam = time.strftime("%H:%M:%S")
    cv2.putText(panel, jam, (CAM_W - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, PUTIH, 1)

    # Status bar bawah
    cv2.rectangle(panel, (0, CAM_H - 55), (CAM_W, CAM_H), ABU2, -1)
    cv2.rectangle(panel, (0, CAM_H - 55), (8, CAM_H), status_warna, -1)  # strip warna
    cv2.putText(panel, "STATUS:", (14, CAM_H - 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, ABU[::- 1], 1)
    cv2.putText(panel, status_teks, (14, CAM_H - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_warna, 2)

    # Info kecil di kanan bawah
    for i, info in enumerate(info_list):
        cv2.putText(panel, info, (CAM_W - 220, CAM_H - 36 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, ABU[::- 1], 1)

    # Border panel
    cv2.rectangle(panel, (0, 0), (CAM_W - 1, CAM_H - 1), (80, 80, 80), 1)

    return panel

# ── Buka Kamera ────────────────────────────────────────────
cap1 = cv2.VideoCapture(KAMERA_PINTU)
cap2 = cv2.VideoCapture(KAMERA_RUANGAN)

if not cap1.isOpened():
    print("ERROR: Kamera 1 tidak terdeteksi!")
    exit()

# ── Setup Jendela Fullscreen ───────────────────────────────
cv2.namedWindow("SIGAP K3", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("SIGAP K3", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Sistem SIGAP aktif! Tekan Q untuk keluar.")

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1:
        cap1 = cv2.VideoCapture(KAMERA_PINTU)
        continue

    # ── Proses Kamera 1 (PPE) ──────────────────────────────
    hasil_ppe    = ppe_model(frame1, conf=0.45, verbose=False)
    frame1_ann   = hasil_ppe[0].plot()
    kelas_ada    = [ppe_model.names[int(b.cls[0])] for b in hasil_ppe[0].boxes]
    apd_kurang   = [k for k in kelas_ada if k in KELAS_TIDAK_LENGKAP]
    apd_lengkap  = [k for k in kelas_ada if k in KELAS_LENGKAP]
    jml_orang1   = sum(1 for k in kelas_ada if k == 'Person')

    if apd_kurang:
        status1       = "DILARANG MASUK"
        warna1        = MERAH
        kurang_str    = ", ".join(set(apd_kurang))
        info1         = [f"Kurang: {kurang_str[:28]}", f"Orang: {jml_orang1}"]
        if time.time() - waktu_alarm_apd > COOLDOWN:
            total_pelanggaran += 1
            bunyikan_alarm("Peringatan! APD tidak lengkap, dilarang masuk!")
            waktu_alarm_apd = time.time()
    elif apd_lengkap:
        status1 = "SILAKAN MASUK"
        warna1  = HIJAU
        info1   = [f"APD Lengkap", f"Orang: {jml_orang1}"]
    else:
        status1 = "Menunggu pekerja..."
        warna1  = ABU
        info1   = ["Tidak ada orang", ""]

    panel1 = buat_panel_kamera(
        frame1_ann,
        "CAM-01",
        "Pintu Masuk | PPE Detection",
        status1, warna1, info1
    )

    # ── Proses Kamera 2 (Fall) ─────────────────────────────
    if ret2:
        pose_hasil = pose_model(frame2, verbose=False)
        frame2_ann = pose_hasil[0].plot()
        ada_jatuh  = False
        jml_orang2 = 0

        if pose_hasil[0].keypoints is not None:
            jml_orang2 = len(pose_hasil[0].keypoints.xy)
            for orang_kps in pose_hasil[0].keypoints.xy:
                kps = orang_kps.cpu().numpy()
                if cek_jatuh(kps):
                    ada_jatuh   = True
                    waktu_jatuh = time.time()
                    if time.time() - waktu_alarm_fall > COOLDOWN:
                        bunyikan_alarm("Darurat! Pekerja jatuh! Hubungi tim keselamatan!")
                        waktu_alarm_fall = time.time()
                    break

        masih_alert = waktu_jatuh and (time.time() - waktu_jatuh < DURASI_ALERT)

        if ada_jatuh or masih_alert:
            status2 = "!! PEKERJA JATUH !!"
            warna2  = MERAH
            info2   = ["Hubungi tim K3!", f"Orang: {jml_orang2}"]
            # Kedip border merah
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame2_ann, (0, 0),
                              (frame2_ann.shape[1], frame2_ann.shape[0]),
                              (0, 0, 255), 10)
        else:
            status2 = "KONDISI AMAN"
            warna2  = HIJAU
            info2   = ["Monitoring aktif", f"Orang: {jml_orang2}"]

        panel2 = buat_panel_kamera(
            frame2_ann,
            "CAM-02",
            "Dalam Ruangan | Fall Detection",
            status2, warna2, info2
        )
    else:
        # Kamera 2 tidak tersedia — tampilkan placeholder
        panel2 = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        teks_tengah(panel2, "CAM-02 TIDAK TERSEDIA", CAM_H // 2,
                    0.7, (100, 100, 100), 1)

    # ── Susun Dashboard ────────────────────────────────────
    # Hitung ukuran layar total
    DASH_W = CAM_W * 2 + 30   # 2 kamera + padding tengah
    HEADER = 60
    FOOTER = 80
    DASH_H = HEADER + CAM_H + FOOTER

    dashboard = np.zeros((DASH_H, DASH_W, 3), dtype=np.uint8)

    # ── Header ─────────────────────────────────────────────
    cv2.rectangle(dashboard, (0, 0), (DASH_W, HEADER), ABU2, -1)
    cv2.line(dashboard, (0, HEADER), (DASH_W, HEADER), (80, 80, 80), 1)

    teks_tengah(dashboard, JUDUL_SISTEM,  22, 0.8, KUNING, 2)
    teks_tengah(dashboard, NAMA_LOKASI,   45, 0.45, (180, 180, 180), 1)

    # Indikator hidup di kanan header
    cv2.circle(dashboard, (DASH_W - 20, 30), 8, HIJAU, -1)
    cv2.putText(dashboard, "LIVE", (DASH_W - 55, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, HIJAU, 1)

    # ── Tempel Panel Kamera ────────────────────────────────
    dashboard[HEADER:HEADER + CAM_H, 0:CAM_W]             = panel1
    dashboard[HEADER:HEADER + CAM_H, CAM_W + 30:DASH_W]   = panel2

    # Garis pemisah tengah
    cx = CAM_W + 15
    cv2.line(dashboard, (cx, HEADER), (cx, HEADER + CAM_H), (80, 80, 80), 2)

    # ── Footer ─────────────────────────────────────────────
    fy = HEADER + CAM_H
    cv2.rectangle(dashboard, (0, fy), (DASH_W, DASH_H), ABU3, -1)
    cv2.line(dashboard, (0, fy), (DASH_W, fy), (80, 80, 80), 1)

    # Hitung uptime
    uptime_det = int(time.time() - waktu_mulai)
    jam_up     = uptime_det // 3600
    menit_up   = (uptime_det % 3600) // 60
    detik_up   = uptime_det % 60
    uptime_str = f"{jam_up:02d}:{menit_up:02d}:{detik_up:02d}"

    # Info footer
    tanggal = time.strftime("%A, %d %B %Y")
    jam_str = time.strftime("%H:%M:%S")

    footer_items = [
        (f"Tanggal : {tanggal}",         20),
        (f"Jam     : {jam_str}",         20 + DASH_W // 4),
        (f"Uptime  : {uptime_str}",      20 + DASH_W // 2),
        (f"Pelanggaran Hari Ini: {total_pelanggaran}", 20 + 3 * DASH_W // 4),
    ]

    for teks, x in footer_items:
        cv2.putText(dashboard, teks, (x, fy + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

    # Label versi
    cv2.putText(dashboard, "K3 AI System | Created by Sufi Anugrah & Abu Yazid Bustomi",
                (DASH_W // 2 - 120, fy + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)

    # ── Tampilkan ──────────────────────────────────────────
    cv2.imshow("SIGAP K3", dashboard)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
print("Sistem SIGAP dimatikan.")
