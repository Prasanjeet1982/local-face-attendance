import uuid
from datetime import datetime
import numpy as np
from sqlalchemy import Column, String, Boolean, DateTime, Float, LargeBinary, Integer, ForeignKey
from sqlalchemy.orm import relationship
from app.db.database import Base


class Employee(Base):
    __tablename__ = "employees"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(64), unique=True, index=True, nullable=False)
    name = Column(String(128), nullable=False)
    department = Column(String(128), nullable=False, default="General")
    embedding = Column(LargeBinary, nullable=False)  # 512D float32 serialized bytes
    created_at = Column(DateTime, default=datetime.now, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    attendance_logs = relationship("AttendanceLog", back_populates="employee", cascade="all, delete-orphan")

    def get_embedding_array(self) -> np.ndarray:
        """Deserialize embedding bytes to a normalized 512D float32 numpy array."""
        if not self.embedding:
            return np.array([], dtype=np.float32)
        arr = np.frombuffer(self.embedding, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr

    def set_embedding_array(self, arr: np.ndarray):
        """Serialize a 512D numpy array to bytes after ensuring float32 and normalization."""
        flat_arr = np.ascontiguousarray(arr, dtype=np.float32).flatten()
        norm = np.linalg.norm(flat_arr)
        if norm > 0:
            flat_arr = flat_arr / norm
        self.embedding = flat_arr.tobytes()

    def to_dict(self):
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "name": self.name,
            "department": self.department,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "is_active": self.is_active
        }


class AttendanceLog(Base):
    __tablename__ = "attendance_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(String(64), ForeignKey("employees.employee_id"), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.now, index=True, nullable=False)
    event_type = Column(String(32), nullable=False)  # CHECK_IN, CHECK_OUT
    confidence_score = Column(Float, nullable=False)
    snapshot_path = Column(String(512), nullable=True)

    employee = relationship("Employee", back_populates="attendance_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "employee_id": self.employee_id,
            "employee_name": self.employee.name if self.employee else self.employee_id,
            "department": self.employee.department if self.employee else "Unknown",
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M:%S") if self.timestamp else None,
            "event_type": self.event_type,
            "confidence_score": round(float(self.confidence_score), 4),
            "snapshot_path": self.snapshot_path
        }
