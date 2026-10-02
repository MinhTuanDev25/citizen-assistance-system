"""Page rendering and OCR. PDF bytes are never passed to the OCR engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.indexing.errors import IndexFailure

OCR_VERSION = "paddleocr-2.9.1-dbnet-crnn-cpu"
DEFAULT_DPI = 150
MAX_DPI = 300
MAX_PIXELS = 4_000_000


class FakeRenderer:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def render(self, _data: bytes, page_number: int, dpi: int):
        self.calls.append(page_number)
        return {"page": page_number, "dpi": dpi}


class PdfiumRenderer:
    """Render one 1-based PDF page to an RGB array. The library index is 0-based."""

    def render(self, data: bytes, page_number: int, dpi: int):
        if dpi < 72 or dpi > MAX_DPI:
            raise IndexFailure("pdf_corrupt")
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise IndexFailure("ocr_failed") from exc
        pdf = pdfium.PdfDocument(data)
        try:
            if page_number < 1 or page_number > len(pdf):
                raise IndexFailure("page_range_invalid")
            page = pdf[page_number - 1]
            scale = dpi / 72
            bitmap = page.render(scale=scale)
            image = bitmap.to_numpy()
        finally:
            pdf.close()
        if image.ndim != 3 or image.shape[2] < 3:
            raise IndexFailure("ocr_failed")
        pixels = int(image.shape[0]) * int(image.shape[1])
        if pixels > MAX_PIXELS:
            raise IndexFailure("pdf_oversized")
        return image[:, :, :3]


class FakeOCR:
    def __init__(self, text: str = "trang scan") -> None:
        self.text = text
        self.calls: list[int] = []
        self.images: list = []
        self.version = "fake-ocr-1"

    def read_page(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> str:
        del timeout_s
        if deadline is not None:
            deadline.check()
        self.calls.append(page_number)
        self.images.append(image)
        return self.text


def load_ocr_manifest(model_dir: str) -> dict:
    path = Path(model_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise IndexFailure("ocr_model_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexFailure("ocr_model_missing") from exc
    if manifest.get("version") != OCR_VERSION:
        raise IndexFailure("ocr_model_missing")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise IndexFailure("ocr_model_missing")
    for name, expected in files.items():
        file_path = path / name
        if not file_path.is_file() or not isinstance(expected, str):
            raise IndexFailure("ocr_model_missing")
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != expected:
            raise IndexFailure("ocr_model_missing")
    return manifest


class PaddleOCR:
    """DBNet + CRNN on CPU. One rendered page image per call. Never downloads models."""

    version = OCR_VERSION

    def __init__(self, model_dir: str) -> None:
        root = Path(model_dir)
        self.manifest = load_ocr_manifest(model_dir)
        try:
            from paddleocr import PaddleOCR as Engine
        except ImportError as exc:
            raise IndexFailure("ocr_model_missing") from exc
        kwargs = {
            "use_angle_cls": False,
            "lang": "vi",
            "use_gpu": False,
            "show_log": False,
            "det_model_dir": str(root / "det"),
            "rec_model_dir": str(root / "rec"),
            "cls_model_dir": str(root / "cls"),
        }
        keys = root / "ppocr_keys_v1.txt"
        if keys.is_file():
            kwargs["rec_char_dict_path"] = str(keys)
        self._engine = Engine(**kwargs)

    def read_page(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> str:
        """OCR one image on the caller thread. The supervisor owns the hard timeout."""
        del timeout_s
        if deadline is not None:
            deadline.check()
        if page_number < 1:
            raise IndexFailure("page_range_invalid")
        try:
            value = self._engine.ocr(image, cls=False)
        except Exception as exc:
            raise IndexFailure("ocr_failed") from exc
        lines: list[str] = []
        for block in value or []:
            for item in block or []:
                if len(item) >= 2 and item[1]:
                    lines.append(str(item[1][0]))
        return "\n".join(lines)
