"""
clinical-image-ai: AI-powered clinical image enhancement pipeline
Usage:
    python main.py --input ./samples --output ./output --report
"""

import argparse
import sys
from pathlib import Path
from pipeline.enhance import process_batch
from pipeline.report import generate_report


def main():
    parser = argparse.ArgumentParser(description="Clinical Image AI Enhancement Pipeline")
    parser.add_argument("--input", required=True, help="Input folder with clinical images")
    parser.add_argument("--output", default="./output", help="Output folder for enhanced images")
    parser.add_argument("--report", action="store_true", help="Generate compliance PDF report")
    parser.add_argument("--guideline", default="ISO-15223", help="Clinical guideline version tag")
    parser.add_argument("--formats", nargs="+", default=["tiff", "png"],
                        choices=["tiff", "png", "jpeg"], help="Export formats")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    images = list(input_path.glob("*.png")) + list(input_path.glob("*.jpg")) + \
             list(input_path.glob("*.jpeg")) + list(input_path.glob("*.tiff")) + \
             list(input_path.glob("*.tif"))

    if not images:
        print(f"[WARN] No images found in {input_path}. Creating demo synthetic image.")
        from pipeline.demo import create_demo_image
        demo_img = create_demo_image(input_path)
        images = [demo_img]

    print(f"[INFO] Found {len(images)} image(s) to process")
    print(f"[INFO] Guideline tag: {args.guideline}")
    print(f"[INFO] Export formats: {args.formats}")
    print()

    results = process_batch(images, output_path, args.formats, args.guideline)

    print(f"\n[DONE] Enhanced {results['processed']} image(s) -> {output_path}")

    if args.report:
        report_path = output_path / "compliance_report.pdf"
        generate_report(results, report_path, args.guideline)
        print(f"[DONE] Compliance report -> {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
