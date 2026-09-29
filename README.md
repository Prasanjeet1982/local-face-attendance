# ⚡ Production Local Face Attendance System

A production-grade, 100% offline, local Face Attendance System running on Python 3.10+. Operates entirely on-premise without external cloud APIs, utilizing InsightFace (ArcFace 512D embeddings + RetinaFace detector), ONNX Runtime (GPU/CPU), OpenCV multi-threaded pipelines, SQLite database storage, and modern dual interfaces (Web Kiosk & PyQt6 Native Desktop GUI).

---

## 🌟 Key Features

* **100% Offline & Private:** Zero external cloud dependencies or API calls. All neural networks and database records stay local.
* **Deep Learning Face Engine:** Powered by InsightFace ONNX models (`buffalo_l` / `buffalo_s`) featuring RetinaFace detection and ArcFace 512-dimensional normalized feature embeddings.
* **Hardware Acceleration with CPU Fallback:** Automatically utilizes `CUDAExecutionProvider` when an NVIDIA GPU and CUDA are available; gracefully falls back to optimized `CPUExecutionProvider`.
* **Passive Anti-Spoofing & Liveness:** Built-in multi-stage passive filter:
  - Laplacian variance focus/blur detection to reject printed low-resolution paper.
  - 2D Fast Fourier Transform (FFT) frequency spectrum texture check to reject digital screen re-broadcasting and Moiré patterns.
  - Color channel distribution check.
* **In-Memory Vector Search:** Blazing fast cosine similarity vector matching in RAM (< 1ms per face) with automatic cache invalidation upon enrollment.
* **Smart Attendance Debounce & Cooldown:**
  - Configurable cooldown buffer (default 10 minutes) prevents duplicate logging when people remain near the camera.
  - Automatic event classification: First detection of the day = `CHECK_IN`; subsequent detections after cooldown = `CHECK_OUT`.
  - Automatic snapshot storage of face crops at check-in/out.
* **Dual Modern Interfaces:**
  1. **Web Kiosk & Admin Dashboard:** Real-time MJPEG live camera stream, live attendance count cards, employee roster table, one-click CSV export, and in-browser camera enrollment.
  2. **PyQt6 Desktop GUI:** Native multi-threaded desktop kiosk window with smooth 30 FPS rendering and zero UI freeze.
  3. **OpenCV High-Speed Kiosk:** Lightweight standalone fullscreen OpenCV feed.

---

## 📁 Repository File Tree

```
face recognition/
├── app/
│   ├── __init__.py
│   ├── config.py                 # Central configuration (thresholds, model name, cooldown)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── logger.py             # Formatted system logging
│   │   └── security.py           # Passive liveness & anti-spoofing algorithms
│   ├── db/
│   │   ├── __init__.py
│   │   ├── database.py           # SQLite connection & session context manager
│   │   └── models.py             # Employee & AttendanceLog SQLAlchemy models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_engine.py        # InsightFace RetinaFace & ArcFace embedding engine
│   │   ├── attendance_manager.py # In-memory RAM cache, cooldown debounce, event rules
│   │   ├── video_stream.py       # Multi-threaded OpenCV capture & inference worker
│   │   └── enrollment.py         # Face quality validator and enrollment service
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py             # REST API (/attendance/today, /export, /enroll, etc.)
│   └── ui/
│       ├── __init__.py
│       ├── desktop_kiosk.py      # PyQt6 Native Desktop Kiosk GUI
│       ├── static/               # Web Kiosk assets
│       │   ├── css/style.css     # Sleek dark-mode glassmorphic styling
│       │   └── js/app.js         # Real-time polling & webcam capture logic
│       └── templates/
│           ├── index.html        # Live Kiosk page with video canvas & check-in sidebar
│           └── dashboard.html    # Management Portal (records, employee directory, export)
├── data/
│   ├── snapshots/                # Face crops captured during check-in/out
│   ├── exports/                  # Generated CSV / Excel attendance reports
│   └── attendance.db             # Local SQLite database
├── enroll.py                     # CLI & webcam interactive enrollment tool
├── main_kiosk.py                 # Primary kiosk launcher (OpenCV or Web mode)
├── desktop_app.py                # Standalone PyQt6 desktop launcher
├── api_server.py                 # Standalone FastAPI & Web Kiosk launcher
├── requirements.txt              # Production Python dependencies
└── README.md                     # Documentation
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites & Virtual Environment

Ensure you have Python 3.10+ installed:

```bash
# Verify Python version
python --version

# Create virtual environment (optional but recommended)
python -m venv venv
.\venv\Scripts\activate   # On Windows
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

#### For NVIDIA GPU Acceleration (Optional)
If you have an NVIDIA GPU with CUDA installed:
```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

---

## 💻 Running the System

### Option A: Web Kiosk & Management Portal (Recommended)
Launches the FastAPI server hosting the high-tech browser Kiosk and Admin Dashboard:

```bash
python main_kiosk.py --mode web
# OR
python api_server.py
```
Open your browser at:
* **Live Video Kiosk:** [http://localhost:8000](http://localhost:8000)
* **Management Dashboard & CSV Export:** [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
* **Interactive API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Option B: Native Desktop Kiosk (OpenCV High-Speed)
Launches a multi-threaded OpenCV desktop window:

```bash
python main_kiosk.py --mode opencv
```
* **Green Box:** Recognized employee with name and ArcFace confidence score.
* **Red Box:** Unknown person or rejected spoof attempt.
* **Onscreen Banner:** Real-time visual feedback: `"CHECK_IN registered for [Name] at HH:MM:SS"`.
* Press `q` or `ESC` to exit.

---

### Option C: Native Desktop GUI (PyQt6)
Launches the full desktop native UI with side-by-side video and live attendance table:

```bash
python desktop_app.py
```

---

## 👤 Enrolling Employees

You can enroll employees through **three convenient methods**:

### 1. Interactive Webcam via CLI (`enroll.py`)
```bash
python enroll.py --id EMP-101 --name "Jane Doe" --dept "Engineering" --source webcam
```
* A live camera window will open with a centering guide circle.
* Position your face inside the circle and press **SPACE** to capture and validate.

### 2. From an Image File or Directory
```bash
# Single image
python enroll.py --id EMP-102 --name "John Smith" --dept "Design" --source "C:/path/to/portrait.jpg"

# Folder containing photos of the employee
python enroll.py --id EMP-103 --name "Alice Wong" --dept "Operations" --source "C:/path/to/photos_folder"
```

### 3. Through the Web UI
1. Navigate to [http://localhost:8000](http://localhost:8000).
2. Click **"+ Enroll Employee"**.
3. Either click **"Use Web Camera"** to snap a photo in your browser, or choose an image file from disk.
4. Click **"Save & Register Face"**. The employee is saved to SQLite and immediately synced to memory.

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `GET /api/system/status` | `GET` | System health, ONNX providers (GPU/CPU), and total active employees. |
| `GET /api/attendance/today` | `GET` | Summary of today's attendance (present count, absent count, detailed logs). |
| `GET /api/attendance/export?date=YYYY-MM-DD` | `GET` | Downloads formatted CSV report for the specified date (or today). |
| `POST /api/employees/enroll` | `POST` | Multipart form endpoint (`employee_id`, `name`, `department`, `image`). |
| `GET /api/employees` | `GET` | Lists all registered employees. |
| `DELETE /api/employees/{employee_id}` | `DELETE` | Removes an employee and purges them from the vector search cache. |
| `GET /api/video/feed` | `GET` | Live multipart/x-mixed-replace MJPEG stream with bounding boxes. |

---

## ⚙️ Configuration & Environment Variables

You can customize parameters in `app/config.py` or via environment variables:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `COOLDOWN_SECONDS` | `600` | Cooldown buffer in seconds (10 min) before registering a second log. |
| `RECOGNITION_THRESHOLD` | `0.45` | Minimum cosine similarity score for face identification. |
| `CAMERA_INDEX` | `0` | Default webcam hardware index. |
| `FACE_MODEL_NAME` | `buffalo_l` | InsightFace model family (`buffalo_l` or lightweight `buffalo_s`). |
| `PORT` | `8000` | Port for the FastAPI server. |
