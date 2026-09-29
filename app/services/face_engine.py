import os
import threading
from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import cv2
import onnxruntime
import insightface
from insightface.app import FaceAnalysis

from app.config import settings
from app.core.logger import logger
from app.core.security import check_passive_liveness


class FaceEngine:
    _instance: Optional['FaceEngine'] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return

        self.model_name = settings.MODEL_NAME
        self.det_size = settings.DET_SIZE
        self.det_threshold = settings.DET_THRESHOLD
        self.rec_threshold = settings.RECOGNITION_THRESHOLD

        # Detect GPU / CPU execution provider
        available_providers = onnxruntime.get_available_providers()
        logger.info(f"Available ONNXRuntime providers: {available_providers}")

        if "CUDAExecutionProvider" in available_providers:
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.ctx_id = 0
            logger.info("Initializing FaceEngine with GPU acceleration (CUDA).")
        else:
            self.providers = ["CPUExecutionProvider"]
            self.ctx_id = -1
            logger.info("Initializing FaceEngine on CPU (GPU provider not detected).")

        self.app = None
        self._init_model()
        self._initialized = True

    def _init_model(self):
        try:
            logger.info(f"Loading InsightFace model '{self.model_name}'...")
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=self.providers,
                root=str(settings.MODELS_DIR)
            )
            self.app.prepare(ctx_id=self.ctx_id, det_size=self.det_size)
            logger.info("FaceAnalysis models successfully initialized and loaded into memory.")
        except Exception as e:
            logger.warning(f"Error loading {self.model_name} with provider {self.providers}: {e}")
            logger.info("Retrying with CPUExecutionProvider and ctx_id=-1...")
            try:
                self.providers = ["CPUExecutionProvider"]
                self.ctx_id = -1
                self.app = FaceAnalysis(
                    name=self.model_name,
                    providers=self.providers,
                    root=str(settings.MODELS_DIR)
                )
                self.app.prepare(ctx_id=-1, det_size=self.det_size)
                logger.info("FaceAnalysis successfully initialized on CPU fallback.")
            except Exception as ex:
                logger.error(f"Failed to initialize FaceAnalysis: {ex}")
                raise RuntimeError(f"Face model failed to initialize: {ex}")

    def detect_faces(self, image_bgr: np.ndarray) -> List[Any]:
        """
        Runs face detection and alignment on the input BGR image.
        Returns a list of InsightFace Face objects containing:
        - bbox: [x1, y1, x2, y2]
        - kps: 5 facial landmarks (eyes, nose, mouth corners)
        - det_score: detection confidence float
        - embedding: 512D ArcFace feature vector
        """
        if self.app is None or image_bgr is None or image_bgr.size == 0:
            return []
        try:
            faces = self.app.get(image_bgr)
            # Filter by detection threshold
            faces = [f for f in faces if f.det_score >= self.det_threshold]
            return faces
        except Exception as e:
            logger.error(f"Exception during face detection: {e}")
            return []

    def extract_embedding(self, image_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extracts 512D normalized embedding, bounding box, and 5-point landmarks
        for the most prominent/largest face in the image.
        Returns: (embedding, bbox, landmarks) or (None, None, None)
        """
        faces = self.detect_faces(image_bgr)
        if not faces:
            return None, None, None

        # If multiple faces exist, pick the largest face by bounding box area
        best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        embedding = best_face.normed_embedding
        if embedding is None:
            raw_emb = best_face.embedding
            norm = np.linalg.norm(raw_emb)
            embedding = raw_emb / norm if norm > 0 else raw_emb

        return embedding.astype(np.float32), best_face.bbox.astype(int), best_face.kps

    def check_liveness(self, face_crop: np.ndarray) -> bool:
        """Passive liveness test against static photo/screen spoofing."""
        is_live, _, _ = check_passive_liveness(face_crop)
        return is_live

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes Cosine Similarity between two 512D normalized vectors."""
        if emb1 is None or emb2 is None:
            return 0.0
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    def match_face(
        self,
        embedding: np.ndarray,
        enrolled_embeddings: Dict[str, np.ndarray],
        threshold: Optional[float] = None
    ) -> Tuple[Optional[str], float]:
        """
        Matches a 512D probe embedding against a dictionary of {employee_id: embedding_array}.
        Returns (best_employee_id, confidence) if confidence >= threshold, else (None, confidence).
        """
        if threshold is None:
            threshold = self.rec_threshold

        if not enrolled_embeddings or embedding is None:
            return None, 0.0

        # Fast vectorised cosine similarity with numpy
        keys = list(enrolled_embeddings.keys())
        matrix = np.array([enrolled_embeddings[k] for k in keys], dtype=np.float32)

        # Ensure normalized
        probe = np.asarray(embedding, dtype=np.float32).flatten()
        probe_norm = np.linalg.norm(probe)
        if probe_norm > 0:
            probe = probe / probe_norm

        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix_norms[matrix_norms == 0] = 1.0
        normed_matrix = matrix / matrix_norms

        similarities = np.dot(normed_matrix, probe)
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])

        if best_score >= threshold:
            return keys[best_idx], best_score
        return None, best_score


# Global singleton instance
face_engine = FaceEngine()
