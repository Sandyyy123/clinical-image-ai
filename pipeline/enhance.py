"""
Core AI enhancement pipeline.
Steps per image:
  1. Load + validate
  2. Denoise (Non-local Means)
  3. CLAHE contrast enhancement
  4. Unsharp mask sharpening
  5. Artifact suppression (morphological closing)
  6. Color space normalization
  7. Export to requested formats
  8. Compute quality metrics (SNR, contrast delta)
"""

import time
from pathlib import Path
from datetime import datetime

try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("[WARN] opencv-python not installed. Running in demo/dry-run mode.")


def _snr(img_array: "np.ndarray") -> float:
    """Signal-to-noise ratio (mean/std of luminance channel)."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(float)
    return float(gray.mean() / (gray.std() + 1e-6))


def _contrast_score(img_array: "np.ndarray") -> float:
    """RMS contrast of grayscale image."""
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY).astype(float)
    return float(gray.std())


def _denoise(img: "np.ndarray") -> "np.ndarray":
    """Non-local Means denoising - effective for clinical photographs."""
    return cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10,
                                           templateWindowSize=7, searchWindowSize=21)


def _clahe(img: "np.ndarray") -> "np.ndarray":
    """CLAHE contrast-limited adaptive histogram equalization per channel."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _sharpen(img: "np.ndarray") -> "np.ndarray":
    """Unsharp masking for detail enhancement."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
    return cv2.addWeighted(img, 1.5, blurred, -0.5, 0)


def _suppress_artifacts(img: "np.ndarray") -> "np.ndarray":
    """Morphological closing to fill small artifact holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


def _normalize_color(img: "np.ndarray") -> "np.ndarray":
    """Normalize color channels to reduce illumination variance."""
    img_float = img.astype(np.float32)
    for c in range(3):
        channel = img_float[:, :, c]
        img_float[:, :, c] = (channel - channel.min()) / (channel.max() - channel.min() + 1e-6) * 255
    return img_float.astype(np.uint8)


def _export(img: "np.ndarray", stem: str, output_dir: Path, formats: list) -> list:
    """Export enhanced image to requested formats."""
    paths = []
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    for fmt in formats:
        if fmt == "tiff":
            p = output_dir / f"{stem}_enhanced.tiff"
            pil_img.save(str(p), format="TIFF", compression="lzw")
        elif fmt == "png":
            p = output_dir / f"{stem}_enhanced.png"
            pil_img.save(str(p), format="PNG", optimize=True)
        elif fmt == "jpeg":
            p = output_dir / f"{stem}_enhanced.jpg"
            pil_img.save(str(p), format="JPEG", quality=95)
        paths.append(str(p))
    return paths


def enhance_image(image_path: Path, output_dir: Path, formats: list, guideline: str) -> dict:
    """Run full enhancement pipeline on a single image. Returns metrics dict."""
    stem = image_path.stem
    result = {
        "file": image_path.name,
        "stem": stem,
        "guideline": guideline,
        "timestamp": datetime.utcnow().isoformat(),
        "status": "ok",
        "exports": [],
        "metrics": {}
    }

    if not CV2_AVAILABLE:
        result["status"] = "demo_mode"
        result["metrics"] = {"snr_before": 12.4, "snr_after": 28.7, "contrast_before": 38.1, "contrast_after": 74.3}
        return result

    try:
        img = cv2.imread(str(image_path))
        if img is None:
            result["status"] = "load_error"
            return result

        snr_before = _snr(img)
        contrast_before = _contrast_score(img)

        img = _denoise(img)
        img = _clahe(img)
        img = _sharpen(img)
        img = _suppress_artifacts(img)
        img = _normalize_color(img)

        snr_after = _snr(img)
        contrast_after = _contrast_score(img)

        result["exports"] = _export(img, stem, output_dir, formats)
        result["metrics"] = {
            "snr_before": round(snr_before, 2),
            "snr_after": round(snr_after, 2),
            "snr_gain_pct": round((snr_after - snr_before) / (snr_before + 1e-6) * 100, 1),
            "contrast_before": round(contrast_before, 2),
            "contrast_after": round(contrast_after, 2),
            "contrast_gain_pct": round((contrast_after - contrast_before) / (contrast_before + 1e-6) * 100, 1),
        }
        result["compliance_pass"] = snr_after > 20 and contrast_after > 40

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def process_batch(images: list, output_dir: Path, formats: list, guideline: str) -> dict:
    """Process a list of image paths and return aggregated results."""
    results = []
    t0 = time.time()

    for i, img_path in enumerate(images, 1):
        print(f"  [{i}/{len(images)}] Processing: {img_path.name} ...", end=" ")
        r = enhance_image(img_path, output_dir, formats, guideline)
        status = r["status"]
        snr_gain = r["metrics"].get("snr_gain_pct", "N/A")
        print(f"status={status}  SNR gain={snr_gain}%")
        results.append(r)

    elapsed = round(time.time() - t0, 1)
    compliant = sum(1 for r in results if r.get("compliance_pass", False))

    return {
        "processed": len(results),
        "compliant": compliant,
        "non_compliant": len(results) - compliant,
        "elapsed_sec": elapsed,
        "guideline": guideline,
        "items": results,
    }
