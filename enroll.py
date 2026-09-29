import argparse
import sys
import os
from typing import Optional
from pathlib import Path
import cv2
import numpy as np

from app.config import settings
from app.core.logger import logger
from app.db.database import init_db
from app.services.attendance_manager import attendance_manager
from app.services.enrollment import validate_and_enroll, EnrollmentError


def capture_from_webcam(camera_index: int = 0) -> Optional[np.ndarray]:
    """Opens a live OpenCV window showing the webcam feed. Press SPACE to capture, ESC to cancel."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera {camera_index}.")
        return None

    print("\n-----------------------------------------------------------")
    print("LIVE ENROLLMENT CAMERA ACTIVE")
    print("1. Center your face directly in good lighting.")
    print("2. Press [SPACE] to capture face.")
    print("3. Press [ESC] or 'q' to cancel.")
    print("-----------------------------------------------------------\n")

    captured_frame = None

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        h, w = frame.shape[:2]
        # Draw centering guide circle
        display_frame = frame.copy()
        cv2.circle(display_frame, (w // 2, h // 2), int(min(w, h) * 0.28), (0, 240, 181), 2)
        cv2.putText(display_frame, "Position Face Inside Circle & Press [SPACE]", (30, 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 240, 181), 2)

        cv2.imshow("Enrollment Camera - Press SPACE to capture", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACE bar
            captured_frame = frame.copy()
            break
        elif key in [27, ord('q')]:  # ESC or q
            print("[INFO] Capture cancelled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_frame


def enroll_from_file_or_dir(employee_id: str, name: str, department: str, path: str):
    target = Path(path)
    if not target.exists():
        print(f"[ERROR] Path does not exist: {path}")
        return

    if target.is_file():
        img = cv2.imread(str(target))
        if img is None:
            print(f"[ERROR] Could not decode image file: {target}")
            return
        _enroll_image(employee_id, name, department, img)
    elif target.is_dir():
        # Directory enrollment
        images = list(target.glob("*.jpg")) + list(target.glob("*.png")) + list(target.glob("*.jpeg"))
        if not images:
            print(f"[WARNING] No image files found in {target}")
            return
        print(f"[INFO] Found {len(images)} images in directory. Enrolling using the highest quality photo...")
        # Try each image until one passes validation
        enrolled = False
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is not None:
                try:
                    res = validate_and_enroll(employee_id, name, department, img)
                    print(f"[SUCCESS] Successfully enrolled {name} using {img_path.name}!")
                    enrolled = True
                    break
                except EnrollmentError as e:
                    print(f"Skipping {img_path.name}: {e}")
        if not enrolled:
            print(f"[ERROR] Could not enroll {name} from any image in {target}.")


def _enroll_image(employee_id: str, name: str, department: str, image_bgr: np.ndarray):
    try:
        result = validate_and_enroll(
            employee_id=employee_id,
            name=name,
            department=department,
            image_bgr=image_bgr
        )
        print("\n==========================================")
        print("🎉 EMPLOYEE ENROLLED SUCCESSFULLY!")
        print(f"ID:                 {result['employee']['employee_id']}")
        print(f"Name:               {result['employee']['name']}")
        print(f"Department:         {result['employee']['department']}")
        print(f"Laplacian Variance: {result['laplacian_variance']}")
        print(f"Liveness Check:     {result['liveness_status']}")
        print(f"Embedding Vector:   {result['embedding_dimensions']} Dimensions")
        print("==========================================\n")
    except EnrollmentError as e:
        print(f"\n[ENROLLMENT ERROR] {e}\n")
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}\n")


def main():
    parser = argparse.ArgumentParser(description="Employee Face Enrollment Tool")
    parser.add_argument("--id", required=True, help="Unique Employee ID (e.g. EMP-101)")
    parser.add_argument("--name", required=True, help="Full Name of Employee")
    parser.add_argument("--dept", default="General", help="Department name")
    parser.add_argument("--source", default="webcam", help="'webcam', image file path, or folder path")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for webcam capture")

    args = parser.parse_args()

    init_db()
    attendance_manager.reload_cache()

    if args.source.lower() == "webcam":
        frame = capture_from_webcam(camera_index=args.camera_index)
        if frame is not None:
            _enroll_image(args.id, args.name, args.dept, frame)
    else:
        enroll_from_file_or_dir(args.id, args.name, args.dept, args.source)


if __name__ == "__main__":
    main()
