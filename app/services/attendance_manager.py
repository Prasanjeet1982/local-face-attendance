import os
import threading
from datetime import datetime, date, timedelta
from typing import Optional, Dict, Tuple, Any
import cv2
import numpy as np

from app.config import settings
from app.core.logger import logger
from app.db.database import get_db_context
from app.db.models import Employee, AttendanceLog
from app.services.face_engine import face_engine


class AttendanceManager:
    _instance: Optional['AttendanceManager'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(AttendanceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        # In-memory caches for fast inference
        self.enrolled_embeddings: Dict[str, np.ndarray] = {}  # {employee_id: 512D numpy array}
        self.employee_meta: Dict[str, Dict[str, Any]] = {}    # {employee_id: {"name": ..., "dept": ...}}
        
        # Debounce / Cooldown tracker: {employee_id: last_logged_datetime}
        self.cooldown_tracker: Dict[str, datetime] = {}
        
        # Daily attendance state: {(employee_id, date): {"check_in": time, "last_event": "CHECK_IN"|"CHECK_OUT"}}
        self.daily_state: Dict[Tuple[str, date], Dict[str, Any]] = {}

        self.cache_lock = threading.RLock()
        self.reload_cache()
        self._load_today_state()
        self._initialized = True

    def reload_cache(self):
        """Loads all active employee embeddings into RAM for real-time vector search."""
        with self.cache_lock:
            try:
                with get_db_context() as db:
                    employees = db.query(Employee).filter(Employee.is_active == True).all() # noqa
                    self.enrolled_embeddings.clear()
                    self.employee_meta.clear()

                    for emp in employees:
                        emb = emp.get_embedding_array()
                        if emb.size == 512:
                            self.enrolled_embeddings[emp.employee_id] = emb
                            self.employee_meta[emp.employee_id] = {
                                "name": emp.name,
                                "department": emp.department,
                                "id": emp.id
                            }

                    logger.info(f"AttendanceManager cache updated: {len(self.enrolled_embeddings)} enrolled active employees.")
            except Exception as e:
                logger.warning(f"Could not load employee cache (database might be initializing): {e}")

    def _load_today_state(self):
        """Pre-populates today's attendance state from SQLite on startup."""
        today = date.today()
        start_of_day = datetime.combine(today, datetime.min.time())
        end_of_day = datetime.combine(today, datetime.max.time())

        try:
            with get_db_context() as db:
                logs = (
                    db.query(AttendanceLog)
                    .filter(AttendanceLog.timestamp >= start_of_day, AttendanceLog.timestamp <= end_of_day)
                    .order_by(AttendanceLog.timestamp.asc())
                    .all()
                )
                for log in logs:
                    key = (log.employee_id, today)
                    if key not in self.daily_state:
                        self.daily_state[key] = {
                            "check_in": log.timestamp,
                            "last_event": log.event_type,
                            "last_time": log.timestamp
                        }
                    else:
                        self.daily_state[key]["last_event"] = log.event_type
                        self.daily_state[key]["last_time"] = log.timestamp

                    self.cooldown_tracker[log.employee_id] = log.timestamp

            logger.info(f"Loaded existing attendance records for {len(self.daily_state)} employees today.")
        except Exception as e:
            logger.warning(f"Could not load today's state: {e}")

    def identify_face(self, embedding: np.ndarray, threshold: Optional[float] = None) -> Tuple[Optional[str], float, Optional[Dict[str, Any]]]:
        """
        Matches embedding against cached enrolled embeddings.
        Returns: (employee_id, confidence, employee_meta) or (None, confidence, None)
        """
        with self.cache_lock:
            emp_id, confidence = face_engine.match_face(
                embedding=embedding,
                enrolled_embeddings=self.enrolled_embeddings,
                threshold=threshold
            )
            meta = self.employee_meta.get(emp_id) if emp_id else None
            return emp_id, confidence, meta

    def record_attendance(
        self,
        employee_id: str,
        confidence: float,
        face_crop: Optional[np.ndarray] = None
    ) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        """
        Evaluates debounce buffer and creates CHECK_IN or CHECK_OUT log entry.
        Returns: (recorded: bool, message: str, details: dict)
        """
        now = datetime.now()
        today = now.date()
        cooldown_window = timedelta(seconds=settings.COOLDOWN_SECONDS)

        with self.cache_lock:
            # 1. Cooldown check
            last_logged = self.cooldown_tracker.get(employee_id)
            if last_logged and (now - last_logged) < cooldown_window:
                remaining = int((cooldown_window - (now - last_logged)).total_seconds())
                emp_name = self.employee_meta.get(employee_id, {}).get("name", employee_id)
                return False, f"Cooldown active for {emp_name} ({remaining}s remaining)", None

            # 2. Determine Event Type (CHECK_IN vs CHECK_OUT)
            state_key = (employee_id, today)
            if state_key not in self.daily_state:
                event_type = "CHECK_IN"
            else:
                last_event = self.daily_state[state_key].get("last_event")
                event_type = "CHECK_OUT" if last_event == "CHECK_IN" else "CHECK_IN"

            # 3. Save snapshot crop if enabled
            snapshot_rel_path = None
            if settings.SAVE_SNAPSHOTS and face_crop is not None and face_crop.size > 0:
                filename = f"{employee_id}_{now.strftime('%Y%m%d_%H%M%S')}_{event_type}.jpg"
                filepath = settings.SNAPSHOTS_DIR / filename
                cv2.imwrite(str(filepath), face_crop)
                snapshot_rel_path = f"snapshots/{filename}"

            # 4. Persist to SQLite
            with get_db_context() as db:
                log_entry = AttendanceLog(
                    employee_id=employee_id,
                    timestamp=now,
                    event_type=event_type,
                    confidence_score=confidence,
                    snapshot_path=snapshot_rel_path
                )
                db.add(log_entry)
                db.flush()
                log_dict = log_entry.to_dict()

            # 5. Update in-memory trackers
            self.cooldown_tracker[employee_id] = now
            if state_key not in self.daily_state:
                self.daily_state[state_key] = {
                    "check_in": now,
                    "last_event": event_type,
                    "last_time": now
                }
            else:
                self.daily_state[state_key]["last_event"] = event_type
                self.daily_state[state_key]["last_time"] = now

            emp_name = self.employee_meta.get(employee_id, {}).get("name", employee_id)
            message = f"{event_type} registered for {emp_name} at {now.strftime('%H:%M:%S')} (Conf: {confidence*100:.1f}%)"
            logger.info(message)

            return True, message, log_dict


attendance_manager = AttendanceManager()
