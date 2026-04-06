# Sistem Monitoring K3 Terintegrasi Kecerdasan Buatan
### untuk Deteksi Real-Time APD dan Insiden Kecelakaan Kerja

> Inovasi K3 berbasis AI Vision menggunakan YOLOv8 dan Computer Vision
> untuk pengawasan keselamatan kerja secara otomatis, real-time, 24 jam.

---

## Fitur Utama

- **PPE Detection** — Deteksi otomatis helm, rompi, dan masker di pintu masuk
- **Fall Detection** — Deteksi pekerja jatuh menggunakan Pose Estimation
- **Alarm Suara** — Peringatan suara otomatis Bahasa Indonesia
- **Dashboard Real-time** — Tampilan monitoring 2 kamera dalam 1 layar
- **Edge Computing** — Berjalan lokal, tanpa internet, tanpa cloud

---

## Teknologi

| Library | Fungsi |
|---|---|
| YOLOv8 (Ultralytics) | Deteksi objek & pose estimation |
| OpenCV | Pemrosesan video real-time |
| PyTorch (CPU) | Backend deep learning |
| espeak-ng | Alarm suara Bahasa Indonesia |
| Python 3.11 | Bahasa pemrograman utama |

---

## Cara Instalasi (Windows)

### 1. Clone Repository
```bash
git clone https://github.com/sufiarh/sistem-monitoring-k3.git
cd sistem-monitoring-k3
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Library
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy
```

### 4. Download Model AI
Download `ppe_model.pt` dari [Releases](https://github.com/sufiarh/sistem-monitoring-k3/releases) 
dan letakkan di folder `models/`

### 5. Jalankan Sistem
```bash
cd src
python k3_dashboard.py
```

---

## Dataset & Model

- **Dataset**: [Construction Site Safety v28](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) — Roboflow Universe (CC BY 4.0)
- **Model PPE**: Download di [Releases](https://github.com/sufiarh/sistem-monitoring-k3/releases)
- **Model Pose**: `yolov8n-pose.pt` otomatis terunduh saat pertama dijalankan

---

## Tim Pengembang

| Nama |
|---|
| Sufi Anugrah |
| Abu Yazid Bustomi |

---

## Kompetisi

Karya ini diajukan untuk **Lomba Inovasi K3 2026**  
DK3P Jawa Timur 2026

---

> *"Selamatkan Pekerja, Wujudkan Budaya K3 Bersama Kecerdasan Buatan."*
