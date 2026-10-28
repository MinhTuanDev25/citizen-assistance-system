"""Page rendering and OCR. PDF bytes are never passed to the OCR engine."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.indexing.errors import IndexFailure

OCR_VERSION = "paddleocr-2.9.1-dbnet-crnn-cpu"
DEFAULT_DPI = 150
MAX_DPI = 300
MAX_PIXELS = 4_000_000


class FakeRenderer:
    def __init__(self, blank: bool = True) -> None:
        self.calls: list[int] = []
        self.blank = blank

    def render(self, _data: bytes, page_number: int, dpi: int):
        self.calls.append(page_number)
        return {"page": page_number, "dpi": dpi, "blank": self.blank}


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
            # Large sheets (A3, posters) would exceed MAX_PIXELS at the OCR DPI:
            # render them smaller instead of failing the whole document.
            width_pt, height_pt = page.get_size()
            if width_pt > 0 and height_pt > 0 and width_pt * height_pt * scale * scale > MAX_PIXELS:
                scale = (MAX_PIXELS / (width_pt * height_pt)) ** 0.5 * 0.999
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


@dataclass(frozen=True)
class OcrPage:
    """OCR text plus word-confidence statistics for the quality gate."""

    text: str
    mean_confidence: float
    low_confidence_ratio: float
    word_count: int


# Tesseract word confidence (0-100) under which a word counts as uncertain.
TESSERACT_LOW_CONFIDENCE = 60.0
# Table rules and scan specks come back as symbol-only tokens ("|", "¬", "„").
_NOISE_TOKEN = re.compile(r"^[^\w]+$", re.UNICODE)
_KEEP_SYMBOL_TOKEN = re.compile(r"^[.,;:()\[\]/%+\-–—\"'“”‘’…!?]+$")


def parse_tesseract_tsv(tsv: str) -> OcrPage:
    """Rebuild reading-order lines from `tesseract ... tsv` and score letter words."""
    lines: dict[tuple[str, str, str, str], list[str]] = {}
    order: list[tuple[str, str, str, str]] = []
    confidences: list[float] = []
    for row in tsv.splitlines()[1:]:
        cols = row.split("\t")
        if len(cols) < 12:
            continue
        word = cols[11].strip()
        try:
            conf = float(cols[10])
        except ValueError:
            continue
        if not word or conf < 0:
            continue
        if _NOISE_TOKEN.match(word) and not _KEEP_SYMBOL_TOKEN.match(word):
            continue
        key = (cols[1], cols[2], cols[3], cols[4])
        if key not in lines:
            lines[key] = []
            order.append(key)
        lines[key].append(word)
        if any(ch.isalpha() for ch in word):
            confidences.append(conf)
    text = "\n".join(" ".join(lines[key]) for key in order)
    if not confidences:
        return OcrPage(text=text, mean_confidence=0.0, low_confidence_ratio=1.0, word_count=0)
    low = sum(1 for conf in confidences if conf < TESSERACT_LOW_CONFIDENCE)
    return OcrPage(
        text=text,
        mean_confidence=sum(confidences) / len(confidences),
        low_confidence_ratio=low / len(confidences),
        word_count=len(confidences),
    )


class TesseractOCR:
    """Tesseract 5 with the `vie` model, run as a child process per page.

    The child stays in the pipeline worker's process group, so cancelling the
    index job kills it. Rendering stays in PdfiumRenderer; only images reach OCR.
    """

    vietnamese_capable = True

    def __init__(self, lang: str = "vie", psm: int = 3, binary: str = "tesseract") -> None:
        self.lang = lang
        self.psm = psm
        self.binary = binary
        try:
            version_out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=10)
            langs_out = subprocess.run([binary, "--list-langs"], capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise IndexFailure("ocr_model_missing") from exc
        first = (version_out.stdout or version_out.stderr).strip().splitlines()
        match = re.match(r"tesseract\s+v?(\S+)", first[0]) if first else None
        langs = langs_out.stdout.strip().splitlines()
        if version_out.returncode != 0 or match is None or lang not in [item.strip() for item in langs[1:]]:
            raise IndexFailure("ocr_model_missing")
        tessdata = re.search(r'"([^"]+)"', langs[0]) if langs else None
        model = Path(tessdata.group(1)) / f"{lang}.traineddata" if tessdata else None
        if model is None or not model.is_file():
            raise IndexFailure("ocr_model_missing")
        digest = hashlib.sha256(model.read_bytes()).hexdigest()[:12]
        self.version = f"tesseract-{match.group(1)}-{lang}-{digest}-psm{psm}"

    def read_page_scored(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> OcrPage:
        if deadline is not None:
            deadline.check()
            timeout_s = min(timeout_s, max(0.5, deadline.remaining()))
        if page_number < 1:
            raise IndexFailure("page_range_invalid")
        ppm = _ppm_bytes(image)
        try:
            proc = subprocess.run(
                [self.binary, "stdin", "stdout", "-l", self.lang, "--psm", str(self.psm), "tsv"],
                input=ppm,
                capture_output=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError("tesseract timeout") from exc
        except OSError as exc:
            raise IndexFailure("ocr_failed") from exc
        if proc.returncode != 0:
            raise IndexFailure("ocr_failed")
        return parse_tesseract_tsv(proc.stdout.decode("utf-8", errors="replace"))

    def read_page(self, image, page_number: int, timeout_s: float = 30, deadline=None) -> str:
        return self.read_page_scored(image, page_number, timeout_s=timeout_s, deadline=deadline).text


def _ppm_bytes(image) -> bytes:
    """Binary PPM (P6) from an RGB uint8 array. Tesseract reads it via Leptonica; no Pillow needed."""
    try:
        height, width, channels = image.shape
    except (AttributeError, ValueError) as exc:
        raise IndexFailure("ocr_failed") from exc
    if channels != 3 or str(getattr(image, "dtype", "")) != "uint8":
        raise IndexFailure("ocr_failed")
    return f"P6\n{width} {height}\n255\n".encode("ascii") + image.tobytes()


VIETNAMESE_REC_PROBE = "ăơư"


def rec_dict_supports_vietnamese(model_dir: str | None) -> bool:
    """True only when the recognition charset includes core Vietnamese letters."""
    if not model_dir:
        return False
    keys = Path(model_dir) / "ppocr_keys_v1.txt"
    if not keys.is_file():
        return False
    try:
        blob = keys.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return all(ch in blob for ch in VIETNAMESE_REC_PROBE)


class FakeOCR:
    def __init__(self, text: str = "trang scan", vietnamese_capable: bool = False) -> None:
        self.text = text
        self.calls: list[int] = []
        self.images: list = []
        self.version = "fake-ocr-1"
        self.vietnamese_capable = vietnamese_capable

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
        self.vietnamese_capable = rec_dict_supports_vietnamese(str(root))
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
