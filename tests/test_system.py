import unittest
from datetime import datetime, timedelta
import numpy as np
import cv2

from app.config import settings
from app.db.database import init_db, get_db_context
from app.db.models import Employee, AttendanceLog
from app.core.security import compute_laplacian_variance, compute_fft_texture_metric, check_passive_liveness
from app.services.face_engine import FaceEngine
from app.services.attendance_manager import AttendanceManager


class TestFaceAttendanceSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        init_db()

    def test_01_database_and_embedding_serialization(self):
        """Tests Employee and AttendanceLog models and 512D vector serialization."""
        emp_id = "TEST_EMP_001"
        test_emb = np.random.randn(512).astype(np.float32)
        test_emb = test_emb / np.linalg.norm(test_emb)

        with get_db_context() as db:
            # Clean up if existed
            db.query(Employee).filter(Employee.employee_id == emp_id).delete()
            
            emp = Employee(
                employee_id=emp_id,
                name="Test Person",
                department="QA Testing"
            )
            emp.set_embedding_array(test_emb)
            db.add(emp)
            db.flush()

            # Retrieve and verify embedding restoration
            saved = db.query(Employee).filter(Employee.employee_id == emp_id).first()
            self.assertIsNotNone(saved)
            self.assertEqual(saved.name, "Test Person")
            
            restored_emb = saved.get_embedding_array()
            self.assertEqual(restored_emb.shape, (512,))
            # Cosine similarity between original and restored should be ~1.0
            similarity = np.dot(test_emb, restored_emb)
            self.assertAlmostEqual(similarity, 1.0, places=4)

    def test_02_security_liveness_metrics(self):
        """Tests blur detection and passive liveness filters."""
        # Sharp synthetic image
        sharp_img = np.zeros((200, 200, 3), dtype=np.uint8)
        for i in range(0, 200, 10):
            cv2.line(sharp_img, (i, 0), (i, 200), (255, 255, 255), 2)
        
        sharp_lap = compute_laplacian_variance(sharp_img)
        self.assertGreater(sharp_lap, 100.0)

        # Extremely blurred synthetic image
        blurred_img = cv2.GaussianBlur(sharp_img, (31, 31), 0)
        blur_lap = compute_laplacian_variance(blurred_img)
        self.assertLess(blur_lap, sharp_lap)

    def test_03_vector_similarity_and_matching(self):
        """Tests in-memory cosine similarity matching."""
        engine = FaceEngine()
        
        # Create 3 synthetic employee embeddings
        v1 = np.random.randn(512).astype(np.float32)
        v1 = v1 / np.linalg.norm(v1)

        v2 = np.random.randn(512).astype(np.float32)
        v2 = v2 / np.linalg.norm(v2)

        v3 = np.random.randn(512).astype(np.float32)
        v3 = v3 / np.linalg.norm(v3)

        roster = {
            "EMP_A": v1,
            "EMP_B": v2,
            "EMP_C": v3
        }

        # Exact match probe
        best_id, conf = engine.match_face(v2, roster, threshold=0.45)
        self.assertEqual(best_id, "EMP_B")
        self.assertGreater(conf, 0.99)

        # Orthogonal/noise vector should fail threshold
        noise = np.random.randn(512).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        matched_id, conf = engine.match_face(noise, roster, threshold=0.85)
        self.assertIsNone(matched_id)

    def test_04_attendance_cooldown_and_events(self):
        """Tests debounce buffer and auto CHECK_IN / CHECK_OUT logic."""
        manager = AttendanceManager()
        emp_id = "TEST_EMP_DEBOUNCE"

        with get_db_context() as db:
            db.query(AttendanceLog).filter(AttendanceLog.employee_id == emp_id).delete()
            db.query(Employee).filter(Employee.employee_id == emp_id).delete()

            emp = Employee(employee_id=emp_id, name="Debounce Tester", department="DevOps")
            dummy_emb = np.ones(512, dtype=np.float32)
            dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)
            emp.set_embedding_array(dummy_emb)
            db.add(emp)

        manager.reload_cache()

        # Clear state for this test employee
        manager.cooldown_tracker.pop(emp_id, None)
        today = datetime.now().date()
        manager.daily_state.pop((emp_id, today), None)

        # 1. First event of day -> MUST be CHECK_IN
        recorded, msg, log = manager.record_attendance(emp_id, confidence=0.92)
        self.assertTrue(recorded)
        self.assertIn("CHECK_IN", msg)

        # 2. Immediate second detection -> MUST trigger cooldown buffer
        recorded2, msg2, log2 = manager.record_attendance(emp_id, confidence=0.94)
        self.assertFalse(recorded2)
        self.assertIn("Cooldown active", msg2)

        # 3. Simulate passage of cooldown window
        manager.cooldown_tracker[emp_id] = datetime.now() - timedelta(seconds=settings.COOLDOWN_SECONDS + 5)
        
        # 4. Subsequent detection -> MUST be CHECK_OUT
        recorded3, msg3, log3 = manager.record_attendance(emp_id, confidence=0.91)
        self.assertTrue(recorded3)
        self.assertIn("CHECK_OUT", msg3)


if __name__ == "__main__":
    unittest.main()
