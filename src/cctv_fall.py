import cv2
import time
from ultralytics import YOLO

print("Loading model pose...")
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

pose_model = YOLO(os.path.join(MODELS_DIR, 'yolov8n-pose.pt'))
# Ganti 0 ke URL RTSP kamera CCTV dalam ruangan nanti
cap = cv2.VideoCapture("http://192.168.43.67:8080/video")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("Gagal konek ke kamera! Pastikan IP Webcam masih running.")
    exit()

print("Kamera terhubung!")

if not cap.isOpened():
    print("ERROR: Kamera tidak terdeteksi!")
    exit()

print("Kamera Dalam Ruangan aktif! Tekan Q untuk keluar.")

# Waktu pertama kali jatuh terdeteksi (untuk alert berkelanjutan)
waktu_jatuh = None
DURASI_ALERT = 5  # detik alert tampil setelah jatuh terdeteksi

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

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pose_hasil = pose_model(frame, verbose=False)
    frame_tampil = pose_hasil[0].plot()
    h, w = frame_tampil.shape[:2]

    ada_jatuh = False

    if pose_hasil[0].keypoints is not None:
        for orang_kps in pose_hasil[0].keypoints.xy:
            kps = orang_kps.cpu().numpy()
            if cek_jatuh(kps):
                ada_jatuh = True
                waktu_jatuh = time.time()
                break

    # Alert tetap tampil beberapa detik setelah jatuh terdeteksi
    masih_alert = waktu_jatuh and (time.time() - waktu_jatuh < DURASI_ALERT)

    if ada_jatuh or masih_alert:
        # Kedipkan border merah
        if int(time.time() * 2) % 2 == 0:
            cv2.rectangle(frame_tampil, (0, 0), (w, h), (0, 0, 255), 8)

        cv2.rectangle(frame_tampil, (0, 0), (w, 65), (0, 0, 200), -1)
        cv2.putText(frame_tampil, "!! DARURAT - PEKERJA JATUH !!",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame_tampil, "Segera hubungi tim keselamatan!",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 1)

    else:
        cv2.rectangle(frame_tampil, (0, 0), (w, 45), (0, 100, 0), -1)
        cv2.putText(frame_tampil, "KONDISI AMAN - Monitoring Aktif",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Timestamp
    waktu_str = time.strftime("%d/%m/%Y %H:%M:%S")
    cv2.putText(frame_tampil, waktu_str, (w - 210, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # Label kamera
    cv2.putText(frame_tampil, "CAM-02 | DALAM RUANGAN", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("K3 - Kamera Dalam Ruangan", frame_tampil)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
