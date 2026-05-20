# clinical-image-ai

AI-powered clinical image enhancement pipeline - batch denoising, CLAHE contrast enhancement, artifact suppression, and compliance reporting for clinical imaging workflows.

```
Raw clinical images
        |
        v
+-------+----------+
| 1. Load + Audit  |   Resolution check, color space validation
+------------------+
        |
        v
+------------------+
| 2. Denoise       |   Non-local Means (NLM) - preserves edge detail
+------------------+
        |
        v
+------------------+
| 3. CLAHE         |   Contrast-Limited Adaptive Histogram Equalization
+------------------+
        |
        v
+------------------+
| 4. Sharpen       |   Unsharp mask for diagnostic detail
+------------------+
        |
        v
+------------------+
| 5. Artifact Fix  |   Morphological closing for small artifact holes
+------------------+
        |
        v
+------------------+
| 6. Normalize     |   Per-channel color normalization
+------------------+
        |
        v
+------------------+
| 7. Export        |   TIFF (LZW) / PNG / JPEG - PS/AI compatible
+------------------+
        |
        v
+------------------+
| 8. Report        |   PDF compliance report with per-image metrics
+------------------+
```

## Tool Stack

| Tool | Purpose |
|------|---------|
| `opencv-python` | Core image processing (NLM denoising, CLAHE, morphology, color ops) |
| `Pillow` | Image I/O, TIFF/PNG/JPEG export, Photoshop-compatible output |
| `numpy` | Array operations, noise metrics, SNR/contrast scoring |
| `scikit-image` | Additional filters (Frangi, Sato for vessel/structure enhancement) |
| `fpdf2` | PDF compliance report generation |

## Quick Start

```bash
pip install -r requirements.txt

# Run on a folder of clinical images
python main.py --input ./your_images --output ./output --report --guideline ISO-15223

# Demo mode (no images needed)
python main.py --input ./samples --output ./output --report
```

## Output

```
output/
  image_name_enhanced.tiff     # LZW-compressed, PS-compatible
  image_name_enhanced.png      # Lossless, web-ready
  compliance_report.pdf        # Per-image metrics + pass/fail tags
```

## Metrics Computed Per Image

- **SNR (Signal-to-Noise Ratio)** - before and after, % gain
- **RMS Contrast** - before and after, % gain
- **Compliance pass/fail** - against configurable thresholds (SNR > 20, contrast > 40)
- **Guideline version tag** - embedded in report metadata

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Folder with input images |
| `--output` | `./output` | Folder for enhanced images |
| `--report` | off | Generate PDF compliance report |
| `--guideline` | `ISO-15223` | Guideline version tag for compliance report |
| `--formats` | `tiff png` | Export formats: `tiff`, `png`, `jpeg` |

## Author

Dr. Sandeep Grover - PhD Data Science (CSIR-IGIB)  
Clinical research background: Charite Berlin, University of Lubeck, University Hospital Tubingen
