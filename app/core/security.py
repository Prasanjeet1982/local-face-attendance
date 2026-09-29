import cv2
import numpy as np
from app.config import settings
from app.core.logger import logger


def compute_laplacian_variance(image_bgr: np.ndarray) -> float:
    """Computes Laplacian variance to measure sharpness/focus."""
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_fft_texture_metric(image_bgr: np.ndarray) -> float:
    """
    Computes frequency spectrum high-frequency ratio via 2D FFT.
    Real human faces have natural skin micro-texture and pore gradients.
    Paper prints and phone screens display either severe high-frequency cutoff
    or sharp regular grid/Moiré peaks.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    if h < 32 or w < 32:
        return 0.0

    # Resize to standard patch for consistent FFT measurement
    resized = cv2.resize(gray, (128, 128))
    f = np.fft.fft2(resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1.0)

    # Calculate center low-frequency energy vs outer ring high-frequency energy
    cy, cx = 64, 64
    r_inner = 16
    r_outer = 48

    y, x = np.ogrid[:128, :128]
    dist_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    inner_mask = dist_from_center <= r_inner
    outer_mask = (dist_from_center > r_inner) & (dist_from_center <= r_outer)

    inner_energy = np.mean(magnitude_spectrum[inner_mask])
    outer_energy = np.mean(magnitude_spectrum[outer_mask])

    if inner_energy <= 0:
        return 0.0

    # Ratio of mid-high frequencies to DC low frequencies
    ratio = (outer_energy / inner_energy) * 10.0
    return float(ratio)


def compute_color_diversity(image_bgr: np.ndarray) -> float:
    """
    Computes standard deviation across HSV saturation and value channels
    to reject black & white printouts or heavily washed out screens.
    """
    if image_bgr is None or image_bgr.size == 0:
        return 0.0
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    sat_std = float(np.std(hsv[:, :, 1]))
    val_std = float(np.std(hsv[:, :, 2]))
    return (sat_std + val_std) / 2.0


def check_passive_liveness(face_crop: np.ndarray) -> tuple[bool, str, dict]:
    """
    Performs passive liveness evaluation:
    1. Size verification (reject tiny crops)
    2. Laplacian focus & blur detection
    3. FFT frequency micro-texture verification
    4. Color variance check
    """
    if not settings.ENABLE_LIVENESS_CHECK:
        return True, "Liveness check disabled", {}

    if face_crop is None or face_crop.size == 0:
        return False, "Empty face crop", {}

    h, w = face_crop.shape[:2]
    if h < settings.MIN_FACE_SIZE or w < settings.MIN_FACE_SIZE:
        return False, f"Face too small ({w}x{h}px)", {"width": w, "height": h}

    laplacian_val = compute_laplacian_variance(face_crop)
    fft_val = compute_fft_texture_metric(face_crop)
    color_var = compute_color_diversity(face_crop)

    metrics = {
        "laplacian": round(laplacian_val, 2),
        "fft_texture": round(fft_val, 2),
        "color_var": round(color_var, 2)
    }

    # 1. Blur / Low-quality screen check
    if laplacian_val < settings.BLUR_LAPLACIAN_THRESHOLD:
        return False, f"Blurry face or low-quality display (Laplacian: {laplacian_val:.1f} < {settings.BLUR_LAPLACIAN_THRESHOLD})", metrics

    # 2. Paper printout or monochrome check
    if color_var < 15.0:
        return False, f"Low color variance / Paper print artifact ({color_var:.1f})", metrics

    # 3. FFT texture check: extremely low indicates flat photo/screen
    if fft_val < settings.FFT_TEXTURE_THRESHOLD:
        return False, f"Abnormal frequency texture (Score: {fft_val:.2f} < {settings.FFT_TEXTURE_THRESHOLD})", metrics

    return True, "Live face verified", metrics
