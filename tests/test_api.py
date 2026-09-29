import unittest
import io
import cv2
import numpy as np
from fastapi.testclient import TestClient
from app.main import app
from app.db.database import init_db, get_db_context
from app.db.models import Employee


class TestAttendanceAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()
        cls.client = TestClient(app)

    def test_01_system_status(self):
        response = self.client.get("/api/system/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("project", data)
        self.assertIn("onnx_providers", data)
        self.assertIn("total_active_employees", data)

    def test_02_attendance_today(self):
        response = self.client.get("/api/attendance/today")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("present_count", data)
        self.assertIn("logs", data)

    def test_03_export_csv(self):
        response = self.client.get("/api/attendance/export")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/csv; charset=utf-8")
        self.assertIn("Employee ID", response.text)

    def test_04_employees_list(self):
        response = self.client.get("/api/employees")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_05_enrollment_validation(self):
        # Empty file test
        response = self.client.post(
            "/api/employees/enroll",
            data={"employee_id": "TEST_ERR", "name": "Error Test", "department": "QA"},
            files={"image": ("blank.jpg", b"", "image/jpeg")}
        )
        self.assertEqual(response.status_code, 400)

    def test_06_html_pages(self):
        res_kiosk = self.client.get("/")
        self.assertEqual(res_kiosk.status_code, 200)
        self.assertIn("VisionPulse", res_kiosk.text)

        res_dash = self.client.get("/dashboard")
        self.assertEqual(res_dash.status_code, 200)
        self.assertIn("Management Portal", res_dash.text)


if __name__ == "__main__":
    unittest.main()
