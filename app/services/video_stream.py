import cv2
import time
import threading
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import numpy as np

from app.config import settings
from app.core.logger import logger
from app.services.face_engine import face_engine
from app.services.attendance_manager import attendance_manager
from app.core.security import check_passive_liveness


class VideoStreamManager:
    _instance: Optional['VideoStreamManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(VideoStreamManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, camera_index: Optional[int] = None):
        if getattr(self, "_initialized", False):
            return

        self.camera_index = camera_index if camera_index is not None else settings.DEFAULT_CAMERA_INDEX
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        
        # Thread safe frame storage
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.frame_lock = threading.Lock()

        # Detection results cache for non-blocking UI rendering
        self.current_detections: List[Dict[str, Any]] = []
        self.detections_lock = threading.Lock()

        # UI Visual Toast / Notification state
        self.notification_text: Optional[str] = None
        self.notification_expiry: float = 0.0
        self.notification_is_success: bool = True

        # Threads
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        self._initialized = True

    def start(self):
        """Starts video capture and inference background workers."""
        if self.running:
            return

        logger.info(f"Opening camera index {self.camera_index}...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            # Try DSHOW on Windows if standard index fails
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)

        if not self.cap.isOpened():
            logger.warning(f"Could not open camera {self.camera_index}. VideoStream will run in simulated/standby mode.")
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, settings.FPS_LIMIT)

        self.running = True

        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True, name="CamCaptureThread")
        self.inference_thread = threading.Thread(target=self._inference_worker, daemon=True, name="FaceInferenceThread")

        self.capture_thread.start()
        self.inference_thread.start()
        logger.info("Video capture and inference pipelines started.")

    def stop(self):
        """Stops video stream threads and releases hardware."""
        self.running = False
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=1.0)

        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
        logger.info("Video pipeline stopped.")

    def _capture_worker(self):
        """Dedicated high-frequency thread to drain OpenCV video buffers without lag."""
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.frame_lock:
                        self.latest_raw_frame = frame
                else:
                    time.sleep(0.01)
            else:
                # Generate modern standby test pattern if no physical camera connected
                standby = self._create_standby_frame()
                with self.frame_lock:
                    self.latest_raw_frame = standby
                time.sleep(0.033)

    def _create_standby_frame(self) -> np.ndarray:
        """Generates a sleek placeholder frame if webcam is in use or offline."""
        img = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
        # Gradient background
        for y in range(settings.FRAME_HEIGHT):
            shade = int(20 + 25 * (y / settings.FRAME_HEIGHT))
            img[y, :] = (shade, shade + 5, shade + 15)

        cv2.putText(img, "LOCAL AI FACE ATTENDANCE SYSTEM", (settings.FRAME_WIDTH // 2 - 320, settings.FRAME_HEIGHT // 2 - 40),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(img, "CAMERA STANDBY / READY FOR STREAM", (settings.FRAME_WIDTH // 2 - 270, settings.FRAME_HEIGHT // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1, cv2.LINE_AA)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, ts, (settings.FRAME_WIDTH // 2 - 100, settings.FRAME_HEIGHT // 2 + 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA)
        return img

    def _inference_worker(self):
        """Asynchronous worker for face detection, liveness, and recognition matching."""
        while self.running:
            raw = None
            with self.frame_lock:
                if self.latest_raw_frame is not None:
                    raw = self.latest_raw_frame.copy()

            if raw is None:
                time.sleep(0.02)
                continue

            try:
                # Detect all faces in frame
                faces = face_engine.detect_faces(raw)
                new_detections = []

                for face in faces:
                    bbox = face.bbox.astype(int)
                    x1 = max(0, bbox[0])
                    y1 = max(0, bbox[1])
                    x2 = min(raw.shape[1], bbox[2])
                    y2 = min(raw.shape[0], bbox[3])

                    face_crop = raw[y1:y2, x1:x2]
                    
                    # 1. Passive Anti-spoofing check
                    is_live, spoof_reason, _ = check_passive_liveness(face_crop)
                    
                    det_data = {
                        "bbox": (x1, y1, x2, y2),
                        "is_live": is_live,
                        "spoof_reason": spoof_reason,
                        "employee_id": None,
                        "name": "Unknown",
                        "department": "",
                        "confidence": 0.0,
                        "is_recognized": False
                    }

                    if is_live:
                        # Extract embedding and match against enrolled employees
                        normed_emb = face.normed_embedding
                        if normed_emb is None:
                            normed_emb = face.embedding / np.linalg.norm(face.embedding)

                        emp_id, conf, meta = attendance_manager.identify_face(normed_emb)
                        if emp_id and meta:
                            det_data["employee_id"] = emp_id
                            det_data["name"] = meta["name"]
                            det_data["department"] = meta["department"]
                            det_data["confidence"] = conf
                            det_data["is_recognized"] = True

                            # Trigger non-blocking attendance registration
                            recorded, msg, log_entry = attendance_manager.record_attendance(
                                employee_id=emp_id,
                                confidence=conf,
                                face_crop=face_crop
                            )
                            if recorded:
                                self.trigger_notification(msg, is_success=True, duration=4.0)

                    new_detections.append(det_data)

                with self.detections_lock:
                    self.current_detections = new_detections

            except Exception as e:
                logger.error(f"Error in inference worker: {e}")

            time.sleep(0.04)  # ~25 FPS inference throttle to preserve CPU/GPU headroom

    def trigger_notification(self, text: str, is_success: bool = True, duration: float = 3.5):
        """Displays an onscreen visual notification toast in the video HUD."""
        self.notification_text = text
        self.notification_expiry = time.time() + duration
        self.notification_is_success = is_success

    def get_annotated_frame(self) -> np.ndarray:
        """Returns the latest video frame with modern bounding boxes, labels, and HUD overlays."""
        with self.frame_lock:
            if self.latest_raw_frame is None:
                return self._create_standby_frame()
            frame = self.latest_raw_frame.copy()

        with self.detections_lock:
            detections = list(self.current_detections)

        # Draw bounding boxes and tags
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            is_recognized = det["is_recognized"]
            is_live = det["is_live"]

            if not is_live:
                # Red Box for Spoof / Fake
                color = (0, 0, 240)  # Red BGR
                label = f"SPOOF / REJECTED: {det['spoof_reason'][:18]}"
            elif is_recognized:
                # Green Box for Recognized Employee
                color = (0, 220, 80)  # Vibrant Green BGR
                conf_pct = int(det["confidence"] * 100)
                label = f"{det['name']} ({conf_pct}%)"
            else:
                # Amber / Orange Box for Unknown Live Face
                color = (0, 140, 255)  # Orange BGR
                label = "Unknown Person"

            # Draw rounded/corner-accented bounding box
            self._draw_corner_rect(frame, (x1, y1, x2, y2), color, thickness=2, corner_len=20)

            # Draw background tag
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.55
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, 1)

            # Tag background
            tag_y1 = max(0, y1 - text_h - 10)
            tag_y2 = y1
            cv2.rectangle(frame, (x1, tag_y1), (x1 + text_w + 16, tag_y2), color, -1)
            cv2.putText(frame, label, (x1 + 8, tag_y2 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Draw HUD Notification Banner if active
        now = time.time()
        if self.notification_text and now < self.notification_expiry:
            self._draw_hud_banner(frame, self.notification_text, self.notification_is_success)

        return frame

    @staticmethod
    def _draw_corner_rect(img, pt, color, thickness=2, corner_len=18):
        x1, y1, x2, y2 = pt
        # Main box (thinner)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        # Highlight corners (thicker)
        t = thickness + 1
        # Top-Left
        cv2.line(img, (x1, y1), (x1 + corner_len, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 + corner_len), color, t)
        # Top-Right
        cv2.line(img, (x2, y1), (x2 - corner_len, y1), color, t)
        cv2.line(img, (x2, y1), (x2, y1 + corner_len), color, t)
        # Bottom-Left
        cv2.line(img, (x1, y2), (x1 + corner_len, y2), color, t)
        cv2.line(img, (x1, y2), (x1, y2 - corner_len), color, t)
        # Bottom-Right
        cv2.line(img, (x2, y2), (x2 - corner_len, y2), color, t)
        cv2.line(img, (x2, y2), (x2, y2 - corner_len), color, t)

    def _draw_hud_banner(self, img: np.ndarray, message: str, is_success: bool):
        h, w = img.shape[:2]
        banner_h = 56
        overlay = img.copy()

        bg_color = (15, 120, 40) if is_success else (20, 20, 160)
        cv2.rectangle(overlay, (20, h - banner_h - 20), (w - 20, h - 20), bg_color, -1)
        
        # 80% opacity blend
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.rectangle(img, (20, h - banner_h - 20), (w - 20, h - 20), (255, 255, 255), 1)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, message, (40, h - 35), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

    def generate_mjpeg_stream(self):
        """Generator yielding MJPEG multipart frames for FastAPI / Web streaming."""
        while True:
            frame = self.get_annotated_frame()
            ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                time.sleep(0.03)
                continue
            
            frame_bytes = jpeg.tobytes()
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
            )
            time.sleep(0.033)  # ~30 FPS


video_stream_manager = VideoStreamManager()
