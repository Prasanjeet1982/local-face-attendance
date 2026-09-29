import sys
import os
import time
import argparse
import cv2

from app.config import settings
from app.core.logger import logger
from app.db.database import init_db
from app.services.attendance_manager import attendance_manager
from app.services.video_stream import video_stream_manager


def run_opencv_kiosk():
    """Runs high-performance OpenCV desktop kiosk with multi-threaded inference."""
    logger.info("Starting Face Attendance OpenCV Kiosk...")
    init_db()
    attendance_manager.reload_cache()
    video_stream_manager.start()

    window_name = f"{settings.PROJECT_NAME} - Native Kiosk"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    print("\n" + "="*60)
    print("LIVE FACE ATTENDANCE KIOSK RUNNING")
    print(" - Green Box: Recognized Employee (Check-In / Out logged)")
    print(" - Red Box:   Unknown Person or Spoof Attempt")
    print(" - Press 'q' or [ESC] to exit.")
    print("="*60 + "\n")

    try:
        while True:
            frame = video_stream_manager.get_annotated_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [27, ord('q'), ord('Q')]:
                break
    except KeyboardInterrupt:
        pass
    finally:
        video_stream_manager.stop()
        cv2.destroyAllWindows()
        logger.info("Kiosk terminated cleanly.")


def run_web_kiosk():
    """Runs the FastAPI server with live Web Kiosk and REST API."""
    import uvicorn
    logger.info(f"Starting Face Attendance Web Kiosk at http://localhost:{settings.API_PORT}")
    uvicorn.run("app.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=False)


def main():
    parser = argparse.ArgumentParser(description="Face Attendance System Kiosk Runner")
    parser.add_argument(
        "--mode",
        choices=["opencv", "web"],
        default="opencv",
        help="Display mode: 'opencv' for native desktop window, 'web' for web kiosk server"
    )
    args = parser.parse_args()

    if args.mode == "web":
        run_web_kiosk()
    else:
        run_opencv_kiosk()


if __name__ == "__main__":
    main()
