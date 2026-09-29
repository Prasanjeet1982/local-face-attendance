import os
from pathlib import Path
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SNAPSHOTS_DIR = DATA_DIR / "snapshots"
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"

# Ensure runtime directories exist
for directory in [DATA_DIR, SNAPSHOTS_DIR, EXPORTS_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class Settings(BaseModel):
    # Project Info
    PROJECT_NAME: str = "Local AI Face Attendance System"
    VERSION: str = "1.0.0"

    # Database & Storage
    DATABASE_URL: str = f"sqlite:///{DATA_DIR / 'attendance.db'}"
    MODELS_DIR: Path = MODELS_DIR
    SNAPSHOTS_DIR: Path = SNAPSHOTS_DIR
    EXPORTS_DIR: Path = EXPORTS_DIR

    # InsightFace Model Configuration
    # 'buffalo_l' contains: detection (RetinaFace ResNet50), recognition (ArcFace 512D)
    # 'buffalo_s' contains: detection (RetinaFace MobileNet), recognition (MobileFaceNet 512D)
    MODEL_NAME: str = os.getenv("FACE_MODEL_NAME", "buffalo_l")
    DET_SIZE: tuple[int, int] = (640, 640)
    DET_THRESHOLD: float = 0.50
    RECOGNITION_THRESHOLD: float = 0.45  # Cosine similarity threshold (ArcFace)

    # Passive Anti-Spoofing & Quality Thresholds
    BLUR_LAPLACIAN_THRESHOLD: float = 80.0     # Minimum variance of Laplacian
    MIN_FACE_SIZE: int = 80                    # Minimum face width/height in pixels
    ENABLE_LIVENESS_CHECK: bool = True
    FFT_TEXTURE_THRESHOLD: float = 8.0         # High frequency ratio for paper/screen spoof detection

    # Attendance Logic & Cooldown
    COOLDOWN_SECONDS: int = int(os.getenv("COOLDOWN_SECONDS", "600"))  # 10 minutes default
    SAVE_SNAPSHOTS: bool = True

    # Camera & Video Stream
    DEFAULT_CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    FRAME_WIDTH: int = 1280
    FRAME_HEIGHT: int = 720
    FPS_LIMIT: int = 30

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = int(os.getenv("PORT", "8000"))


settings = Settings()
