"""Build an image-only PDF. The font is used at generation time and is not stored in the file."""

from __future__ import annotations

import zlib
from pathlib import Path

LINES = ("GIAY KHAI SINH", "GIA LAI", "SO 123")
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu-sans/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
)


def render_lines(lines: tuple[str, ...], font_path: str | None = None, size: int = 72) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    chosen = font_path
    if chosen is None:
        chosen = next((path for path in FONT_CANDIDATES if Path(path).is_file()), None)
    if chosen is None:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(chosen, size)
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1), "white"))
    widths = [probe.textbbox((0, 0), line, font=font)[2] for line in lines]
    height = size + 16
    image = Image.new("RGB", (max(widths) + 80, height * len(lines) + 40), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((40, 20 + index * height), line, fill="black", font=font)
    return _png(image)


def image_only_pdf(png: bytes) -> bytes:
    """One page whose only content is the PNG. There is no text drawing operator."""
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(png)).convert("RGB")
    width, height = image.size
    raw = zlib.compress(image.tobytes())
    content = f"q {width} 0 0 {height} 0 0 cm /Im Do Q".encode()
    objects = [
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Count 1 /Kids [3 0 R] >>\nendobj\n",
        (
            f"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
            f"/Resources << /XObject << /Im 4 0 R >> >> /Contents 5 0 R >>\nendobj\n"
        ).encode(),
        (
            f"4 0 obj\n<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {len(raw)} >>\nstream\n"
        ).encode()
        + raw
        + b"\nendstream\nendobj\n",
        f"5 0 obj\n<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream\nendobj\n",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for item in objects:
        offsets.append(len(out))
        out.extend(item)
    xref_at = len(out)
    out.extend(f"xref\n0 {len(offsets)}\n".encode())
    out.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        out.extend(f"{offset:010d} 00000 n \n".encode())
    out.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode()
    )
    return bytes(out)


def _png(image) -> bytes:
    import io

    handle = io.BytesIO()
    image.save(handle, format="PNG")
    return handle.getvalue()


def build_scan_pdf(lines: tuple[str, ...] = LINES) -> bytes:
    return image_only_pdf(render_lines(lines))
