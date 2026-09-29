import io
import csv
from datetime import datetime, date
from typing import Optional, List
import cv2
import numpy as np
import onnxruntime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status
from fastapi.responses import StreamingResponse, Response, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.config import settings
from app.core.logger import logger
from app.db.database import get_db
from app.db.models import Employee, AttendanceLog
from app.services.attendance_manager import attendance_manager
from app.services.enrollment import validate_and_enroll, EnrollmentError
from app.services.video_stream import video_stream_manager

api_router = APIRouter()


@api_router.get("/system/status")
def get_system_status(db: Session = Depends(get_db)):
    """System health, hardware providers, and database stats."""
    total_employees = db.query(Employee).filter(Employee.is_active == True).count() # noqa
    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    today_records = db.query(AttendanceLog).filter(AttendanceLog.timestamp >= start_of_day).count()
    unique_present_today = (
        db.query(func.count(func.distinct(AttendanceLog.employee_id)))
        .filter(AttendanceLog.timestamp >= start_of_day)
        .scalar()
    )

    return {
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "onnx_providers": onnxruntime.get_available_providers(),
        "model_name": settings.MODEL_NAME,
        "recognition_threshold": settings.RECOGNITION_THRESHOLD,
        "cooldown_seconds": settings.COOLDOWN_SECONDS,
        "total_active_employees": total_employees,
        "total_present_today": unique_present_today or 0,
        "total_logs_today": today_records,
        "stream_running": video_stream_manager.running
    }


@api_router.get("/attendance/today")
def get_today_attendance(db: Session = Depends(get_db)):
    """Returns today's attendance summary and chronologically ordered log entries."""
    today = date.today()
    start_of_day = datetime.combine(today, datetime.min.time())
    end_of_day = datetime.combine(today, datetime.max.time())

    logs = (
        db.query(AttendanceLog)
        .filter(AttendanceLog.timestamp >= start_of_day, AttendanceLog.timestamp <= end_of_day)
        .order_by(AttendanceLog.timestamp.desc())
        .all()
    )

    total_active = db.query(Employee).filter(Employee.is_active == True).count() # noqa
    present_ids = set(l.employee_id for l in logs)

    return {
        "date": today.strftime("%Y-%m-%d"),
        "total_active_employees": total_active,
        "present_count": len(present_ids),
        "absent_count": max(0, total_active - len(present_ids)),
        "logs": [l.to_dict() for l in logs]
    }


@api_router.get("/attendance/export")
def export_attendance_csv(
    date_str: Optional[str] = Query(None, alias="date", description="Target date in YYYY-MM-DD format"),
    db: Session = Depends(get_db)
):
    """Generates a downloadable CSV report of attendance records for the given date or today."""
    if date_str:
        try:
            target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Expected YYYY-MM-DD.")
    else:
        target_date = date.today()

    start_time = datetime.combine(target_date, datetime.min.time())
    end_time = datetime.combine(target_date, datetime.max.time())

    logs = (
        db.query(AttendanceLog)
        .filter(AttendanceLog.timestamp >= start_time, AttendanceLog.timestamp <= end_time)
        .order_by(AttendanceLog.timestamp.asc())
        .all()
    )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Log ID", "Employee ID", "Employee Name", "Department", "Event Type", "Timestamp", "Confidence Score", "Snapshot Path"])

    for log in logs:
        writer.writerow([
            log.id,
            log.employee_id,
            log.employee.name if log.employee else "N/A",
            log.employee.department if log.employee else "N/A",
            log.event_type,
            log.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            f"{log.confidence_score:.4f}",
            log.snapshot_path or "None"
        ])

    csv_data = output.getvalue()
    filename = f"attendance_report_{target_date.strftime('%Y%m%d')}.csv"

    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@api_router.post("/employees/enroll", status_code=status.HTTP_201_CREATED)
async def enroll_employee(
    employee_id: str = Form(...),
    name: str = Form(...),
    department: str = Form("General"),
    image: UploadFile = File(...)
):
    """
    Multipart image upload enrollment endpoint.
    Performs face quality checks, extracts ArcFace 512D vector, and saves employee.
    """
    contents = await image.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Uploaded file could not be decoded as an image.")

    try:
        result = validate_and_enroll(
            employee_id=employee_id.strip(),
            name=name.strip(),
            department=department.strip(),
            image_bgr=image_bgr,
            overwrite_existing=True
        )
        return result
    except EnrollmentError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error in employee enrollment: {e}")
        raise HTTPException(status_code=500, detail=f"Enrollment failure: {str(e)}")


@api_router.get("/employees")
def list_employees(db: Session = Depends(get_db)):
    """Lists all enrolled employees."""
    employees = db.query(Employee).order_by(Employee.created_at.desc()).all()
    return [e.to_dict() for e in employees]


@api_router.delete("/employees/{employee_id}")
def delete_employee(employee_id: str, db: Session = Depends(get_db)):
    """Deletes or deactivates an employee and clears in-memory cache."""
    emp = db.query(Employee).filter(Employee.employee_id == employee_id).first()
    if not emp:
        raise HTTPException(status_code=404, detail="Employee not found")
    
    db.delete(emp)
    db.commit()
    attendance_manager.reload_cache()
    return {"success": True, "message": f"Employee {employee_id} deleted successfully."}


@api_router.get("/video/feed")
def video_feed():
    """Live MJPEG video stream with bounding boxes and HUD recognition banner."""
    if not video_stream_manager.running:
        video_stream_manager.start()
    return StreamingResponse(
        video_stream_manager.generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
