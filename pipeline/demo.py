"""
Creates a synthetic clinical-style demo image when no input images are provided.
Simulates a noisy dermoscopy photograph for pipeline demonstration.
"""

from pathlib import Path


def create_demo_image(output_dir: Path) -> Path:
    """Generate a synthetic noisy clinical image for demo purposes."""
    try:
        import numpy as np
        from PIL import Image, ImageFilter, ImageDraw

        width, height = 512, 512
        rng = np.random.default_rng(42)

        # Simulate skin-tone background
        base = np.full((height, width, 3), [180, 140, 110], dtype=np.uint8)

        # Add a simulated lesion
        img = Image.fromarray(base)
        draw = ImageDraw.Draw(img)
        draw.ellipse([180, 180, 330, 310], fill=(100, 70, 60))
        draw.ellipse([200, 195, 300, 280], fill=(80, 50, 45))

        # Add Gaussian noise to simulate clinical camera noise
        img_array = np.array(img).astype(np.float32)
        noise = rng.normal(0, 25, img_array.shape)
        noisy = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img_noisy = Image.fromarray(noisy)

        # Slight blur to simulate motion/focus artifact
        img_noisy = img_noisy.filter(ImageFilter.GaussianBlur(radius=1.2))

        output_dir.mkdir(parents=True, exist_ok=True)
        demo_path = output_dir / "demo_clinical_image.png"
        img_noisy.save(str(demo_path))
        print(f"[DEMO] Synthetic clinical image created: {demo_path}")
        return demo_path

    except ImportError:
        # Ultra-minimal fallback: tiny 4x4 placeholder
        from pathlib import Path
        demo_path = output_dir / "demo_placeholder.png"
        # Write minimal valid PNG bytes
        import struct, zlib
        def png_chunk(name, data):
            c = zlib.crc32(name + data) & 0xFFFFFFFF
            return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)
        w, h = 4, 4
        raw = b"\x00" + b"\x80\x60\x50" * w
        raw_rows = raw * h
        compressed = zlib.compress(raw_rows)
        header = b"\x89PNG\r\n\x1a\n"
        ihdr = png_chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        idat = png_chunk(b"IDAT", compressed)
        iend = png_chunk(b"IEND", b"")
        demo_path.write_bytes(header + ihdr + idat + iend)
        print(f"[DEMO] Minimal placeholder image created: {demo_path}")
        return demo_path
