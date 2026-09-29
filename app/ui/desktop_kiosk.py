import sys
import cv2
from datetime import datetime
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QFont
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLineEdit, QDialog, QMessageBox
)

from app.config import settings
from app.db.database import get_db_context, init_db
from app.db.models import Employee, AttendanceLog
from app.services.video_stream import video_stream_manager
from app.services.attendance_manager import attendance_manager
from app.services.enrollment import validate_and_enroll, EnrollmentError


class VideoThread(QThread):
    frame_ready = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        video_stream_manager.start()
        while self.running:
            frame = video_stream_manager.get_annotated_frame()
            if frame is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_frame.shape
                bytes_per_line = ch * w
                qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(qt_img)
            self.msleep(30)

    def stop(self):
        self.running = False
        self.wait(1000)


class DesktopKioskWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{settings.PROJECT_NAME} - Native Kiosk")
        self.resize(1360, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #0b0f19; }
            QLabel { color: #f0f4f8; font-family: Segoe UI, sans-serif; }
            QTableWidget {
                background-color: #121929;
                border: 1px solid #232f48;
                border-radius: 8px;
                color: #e2e8f0;
                gridline-color: #1b2438;
            }
            QHeaderView::section {
                background-color: #1a2236;
                color: #8fa0be;
                padding: 6px;
                border: none;
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #00f0b5, stop:1 #00ab80);
                color: #061713;
                font-weight: bold;
                border-radius: 6px;
                padding: 10px 18px;
                font-size: 13px;
            }
            QPushButton:hover {
                background: #00ffd0;
            }
        """)

        # Main Layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Left Column: Video Screen
        left_box = QVBoxLayout()
        title_label = QLabel("⚡ Live Recognition & Anti-Spoofing Feed")
        title_label.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        left_box.addWidget(title_label)

        self.video_label = QLabel("Initializing Video Engine...")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 560)
        self.video_label.setStyleSheet("background-color: #000; border-radius: 12px; border: 1px solid #232f48;")
        left_box.addWidget(self.video_label, 1)

        main_layout.addLayout(left_box, 3)

        # Right Column: Real-time Attendance & Controls
        right_box = QVBoxLayout()
        right_box.setSpacing(12)

        header_rt = QLabel("📋 Today's Attendance Logs")
        header_rt.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        right_box.addWidget(header_rt)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Time", "ID", "Name", "Event"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        right_box.addWidget(self.table, 1)

        self.btn_enroll = QPushButton("➕ Enroll New Employee")
        self.btn_enroll.clicked.connect(self.open_enroll_dialog)
        right_box.addWidget(self.btn_enroll)

        self.btn_refresh = QPushButton("🔄 Refresh Logs")
        self.btn_refresh.setStyleSheet("background: #1e293b; color: #fff;")
        self.btn_refresh.clicked.connect(self.load_attendance_logs)
        right_box.addWidget(self.btn_refresh)

        main_layout.addLayout(right_box, 2)

        # Start Video Thread
        self.video_thread = VideoThread()
        self.video_thread.frame_ready.connect(self.update_video_frame)
        self.video_thread.start()

        self.load_attendance_logs()

    def update_video_frame(self, qt_img: QImage):
        pixmap = QPixmap.fromImage(qt_img)
        scaled = pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video_label.setPixmap(scaled)

    def load_attendance_logs(self):
        with get_db_context() as db:
            today_start = datetime.combine(datetime.now().date(), datetime.min.time())
            logs = (
                db.query(AttendanceLog)
                .filter(AttendanceLog.timestamp >= today_start)
                .order_by(AttendanceLog.timestamp.desc())
                .limit(20)
                .all()
            )

            self.table.setRowCount(len(logs))
            for row, log in enumerate(logs):
                t_str = log.timestamp.strftime("%H:%M:%S")
                self.table.setItem(row, 0, QTableWidgetItem(t_str))
                self.table.setItem(row, 1, QTableWidgetItem(log.employee_id))
                name = log.employee.name if log.employee else log.employee_id
                self.table.setItem(row, 2, QTableWidgetItem(name))
                self.table.setItem(row, 3, QTableWidgetItem(log.event_type))

    def open_enroll_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Enroll Employee")
        dialog.setFixedWidth(400)
        dialog.setStyleSheet("background-color: #121929; color: #fff;")

        layout = QVBoxLayout(dialog)

        id_input = QLineEdit()
        id_input.setPlaceholderText("Employee ID (e.g. EMP-001)")
        layout.addWidget(QLabel("Employee ID:"))
        layout.addWidget(id_input)

        name_input = QLineEdit()
        name_input.setPlaceholderText("Full Name")
        layout.addWidget(QLabel("Full Name:"))
        layout.addWidget(name_input)

        dept_input = QLineEdit()
        dept_input.setPlaceholderText("Department (e.g. R&D)")
        layout.addWidget(QLabel("Department:"))
        layout.addWidget(dept_input)

        snap_btn = QPushButton("📸 Capture Face from Live Webcam")
        
        def on_snap():
            emp_id = id_input.text().strip()
            name = name_input.text().strip()
            dept = dept_input.text().strip() or "General"

            if not emp_id or not name:
                QMessageBox.warning(dialog, "Missing Fields", "Please enter Employee ID and Name.")
                return

            # Grab current raw frame from video stream
            raw_frame = None
            with video_stream_manager.frame_lock:
                if video_stream_manager.latest_raw_frame is not None:
                    raw_frame = video_stream_manager.latest_raw_frame.copy()

            if raw_frame is None:
                QMessageBox.critical(dialog, "Error", "Camera frame not available.")
                return

            try:
                validate_and_enroll(
                    employee_id=emp_id,
                    name=name,
                    department=dept,
                    image_bgr=raw_frame
                )
                QMessageBox.information(dialog, "Success", f"Enrolled {name} ({emp_id}) successfully!")
                dialog.accept()
                self.load_attendance_logs()
            except EnrollmentError as e:
                QMessageBox.critical(dialog, "Enrollment Failed", str(e))
            except Exception as ex:
                QMessageBox.critical(dialog, "Error", str(ex))

        snap_btn.clicked.connect(on_snap)
        layout.addWidget(snap_btn)
        dialog.exec()

    def closeEvent(self, event):
        self.video_thread.stop()
        video_stream_manager.stop()
        event.accept()


def run_desktop_app():
    init_db()
    attendance_manager.reload_cache()
    app = QApplication(sys.argv)
    window = DesktopKioskWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_desktop_app()
