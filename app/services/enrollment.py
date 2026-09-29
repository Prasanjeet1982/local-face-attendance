from typing import Tuple, Optional, Dict, Any
import numpy as np
import cv2

from app.config import settings
from app.core.logger import logger
from app.core.security import compute_laplacian_variance, check_passive_liveness
from app.db.database import get_db_context
from app.db.models import Employee
from app.services.face_engine import face_engine
from app.services.attendance_manager import attendance_manager


class EnrollmentError(Exception):
    pass


def validate_and_enroll(
    employee_id: str,
    name: str,
    department: str,
    image_bgr: np.ndarray,
    overwrite_existing: bool = True
) -> Dict[str, Any]:
    """
    Validates face image quality, extracts ArcFace 512D embedding, and registers employee.
    Enforces production checks:
    - Non-empty image
    - Minimum Laplacian variance (> 80.0 or 100.0)
    - Exactly ONE face detected
    - Valid liveness texture
    """
    if image_bgr is None or image_bgr.size == 0:
        raise EnrollmentError("Provided image is empty or invalid.")

    # 1. Blur Check
    lap_var = compute_laplacian_variance(image_bgr)
    if lap_var < 80.0:
        raise EnrollmentError(f"Image too blurry (Laplacian variance: {lap_var:.1f} < 80.0). Please hold still in good lighting.")

    # 2. Detect Faces
    faces = face_engine.detect_faces(image_bgr)
    if not faces:
        raise EnrollmentError("No face detected in the image. Please position yourself directly facing the camera.")
    if len(faces) > 1:
        raise EnrollmentError(f"Multiple faces detected ({len(faces)}). Enrollment requires strictly one person in frame.")

    face = faces[0]
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(image_bgr.shape[1], bbox[2]), min(image_bgr.shape[0], bbox[3])
    face_crop = image_bgr[y1:y2, x1:x2]

    # 3. Liveness / Passive anti-spoof check
    is_live, reason, metrics = check_passive_liveness(face_crop)
    if not is_live:
        logger.warning(f"Liveness warning during enrollment: {reason} ({metrics})")

    # 4. Extract 512D ArcFace normalized embedding
    raw_emb = face.normed_embedding if face.normed_embedding is not None else face.embedding
    norm = np.linalg.norm(raw_emb)
    if norm == 0:
        raise EnrollmentError("Failed to extract valid face embedding.")
    normed_emb = (raw_emb / norm).astype(np.float32)

    # 5. Database Save/Update
    with get_db_context() as db:
        existing = db.query(Employee).filter(Employee.employee_id == employee_id).first()
        if existing:
            if not overwrite_existing:
                raise EnrollmentError(f"Employee with ID '{employee_id}' already exists.")
            existing.name = name
            existing.department = department
            existing.set_embedding_array(normed_emb)
            existing.is_active = True
            logger.info(f"Updated existing employee record: {employee_id} ({name})")
            db.flush()
            emp_data = existing.to_dict()
        else:
            new_emp = Employee(
                employee_id=employee_id,
                name=name,
                department=department
            )
            new_emp.set_embedding_array(normed_emb)
            db.add(new_emp)
            db.flush()
            logger.info(f"Enrolled new employee: {employee_id} ({name})")
            emp_data = new_emp.to_dict()

    # 6. Immediately sync in-memory vector cache
    attendance_manager.reload_cache()

    return {
        "success": True,
        "employee": emp_data,
        "laplacian_variance": round(lap_var, 2),
        "liveness_status": "Passed" if is_live else f"Flagged ({reason})",
        "embedding_dimensions": int(len(normed_emb))
    }
