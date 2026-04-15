import cv2
from ultralytics import YOLO

print("Loading model PPE...")
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

ppe_model = YOLO(os.path.join(MODELS_DIR, 'ppe_model.pt'))
# Ganti 0 ke URL RTSP kamera CCTV nanti
# Contoh RTSP: 'rtsp://admin:password@192.168.1.100:554/stream'
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Kamera tidak terdeteksi!")
    exit()

print("Kamera Pintu Masuk aktif! Tekan Q untuk keluar.")

KELAS_TIDAK_LENGKAP = ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']
KELAS_LENGKAP       = ['Hardhat', 'Safety Vest', 'Mask']

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hasil = ppe_model(frame, conf=0.45, verbose=False)

    kelas_terdeteksi = [ppe_model.names[int(b.cls[0])] for b in hasil[0].boxes]

    apd_kurang  = [k for k in kelas_terdeteksi if k in KELAS_TIDAK_LENGKAP]
    apd_lengkap = [k for k in kelas_terdeteksi if k in KELAS_LENGKAP]

    frame_tampil = hasil[0].plot()
    h, w = frame_tampil.shape[:2]

    if apd_kurang:
        # Merah - tidak boleh masuk
        cv2.rectangle(frame_tampil, (0, 0), (w, 55), (0, 0, 180), -1)
        cv2.putText(frame_tampil, "DILARANG MASUK - APD TIDAK LENGKAP",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Tampilkan APD yang kurang
        kurang_str = "Kurang: " + ", ".join(set(apd_kurang))
        cv2.putText(frame_tampil, kurang_str,
                    (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)

    elif apd_lengkap:
        # Hijau - boleh masuk
        cv2.rectangle(frame_tampil, (0, 0), (w, 55), (0, 130, 0), -1)
        cv2.putText(frame_tampil, "SILAKAN MASUK - APD LENGKAP",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)

    else:
        # Abu - tidak ada orang
        cv2.rectangle(frame_tampil, (0, 0), (w, 55), (60, 60, 60), -1)
        cv2.putText(frame_tampil, "Menunggu pekerja...",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 200, 200), 2)

    # Label kamera
    cv2.putText(frame_tampil, "CAM-01 | PINTU MASUK", (w - 230, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("K3 - Kamera Pintu Masuk", frame_tampil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
