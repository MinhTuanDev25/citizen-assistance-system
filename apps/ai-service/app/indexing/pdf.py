"""Read a PDF from bytes already loaded from internal object storage."""

from __future__ import annotations

import io
import re

from app.indexing.errors import IndexFailure
from app.indexing.text import normalize_text

NATIVE_TEXT_MIN_CHARS = 24
# Identity-H ToUnicode on the BTP PDF maps many live CIDs to U+0020, so pypdf
# keeps a long string of broken syllables ("đ ng ký"). Length alone is not a
# quality signal. A high share of 1-letter alphabetic tokens is.
NATIVE_SINGLE_LETTER_MIN_TOKENS = 20
NATIVE_SINGLE_LETTER_MAX_RATIO = 0.18
# Short headings on the same PDF still show leftover letters ("đ ng") before
# the 20-token ratio can fire. ASCII section marks like "A." are not leftovers.
NATIVE_SHORT_FRAGMENT_MIN_TOKENS = 3
# Recovery may not drop most of the Vietnamese letters the native page already
# kept. This is relative to that page, not a global diacritic density rule.
RECOVERY_MIN_NATIVE_VIET = 20
RECOVERY_KEEP_RATIO = 0.5
BLANK_INK_RATIO_MAX = 0.008
# Isolated specks are noise. A consecutive dark run is a stroke, even when the
# ink ratio stays under BLANK_INK_RATIO_MAX (a 128px line on 128×128 is 0.78%).
BLANK_INK_RUN_MIN = 5
OCR_MIN_VIET_LETTERS = 3
_ALPHA_TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
# a) b) đ) at the start of an item are list marks, not CID leftovers.
_ENUM_MARK = re.compile(r"(?m)(^[ \t]*|[\s;]+)[a-zA-ZđĐ]\)[ \t]+")
# Standalone Vietnamese 1-letter tokens are vowels/particles (ở, Ý, a), not CID leftovers.
_VIET_VOWELS = frozenset(
    "aáàảãạăắằẳẵặâấầẩẫậAÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬ"
    "eéèẻẽẹêếềểễệEÉÈẺẼẸÊẾỀỂỄỆ"
    "iíìỉĩịIÍÌỈĨỊ"
    "oóòỏõọôốồổỗộơớờởỡợOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢ"
    "uúùủũụưứừửữựUÚÙỦŨỤƯỨỪỬỮỰ"
    "yýỳỷỹỵYÝỲỶỸỴ"
)


def page_bounds(page_range: str | None, page_count: int) -> tuple[int, int]:
    if page_range is None or page_range == "":
        return 1, page_count
    if "-" in page_range:
        start_s, end_s = page_range.split("-", 1)
        start, end = int(start_s), int(end_s)
    else:
        start = end = int(page_range)
    if start < 1 or end > page_count or start > end:
        raise IndexFailure("page_range_invalid")
    return start, end


def _is_section_mark(token: str) -> bool:
    return len(token) == 1 and token.isascii() and token.isupper()


def _is_cid_singleton(token: str) -> bool:
    """True for a 1-letter leftover such as đ in 'đ ng', not ở / Ý / a)."""
    if len(token) != 1 or not token.isalpha():
        return False
    if _is_section_mark(token) or token in _VIET_VOWELS:
        return False
    return True


def _cid_space_fragmented(tokens: list[str]) -> bool:
    singles = [token for token in tokens if _is_cid_singleton(token)]
    if len(tokens) >= NATIVE_SINGLE_LETTER_MIN_TOKENS:
        return len(singles) / len(tokens) >= NATIVE_SINGLE_LETTER_MAX_RATIO
    if len(tokens) >= NATIVE_SHORT_FRAGMENT_MIN_TOKENS:
        # Short headings use the BTP leftover đ, not ASCII list marks b) c).
        return any(not token.isascii() for token in singles)
    return False


def _without_enum_marks(text: str) -> str:
    return _ENUM_MARK.sub(r"\1", text)


def native_text_usable(text: str, min_chars: int = NATIVE_TEXT_MIN_CHARS) -> bool:
    """Keep native text only when it is long enough and not CID-space-garbled."""
    if len(text) < min_chars:
        return False
    tokens = _ALPHA_TOKEN.findall(_without_enum_marks(text))
    return not _cid_space_fragmented(tokens)


def page_requires_recovery(text: str, min_chars: int = NATIVE_TEXT_MIN_CHARS) -> bool:
    """True when native text exists but is not safe to index. Empty scan pages are not required."""
    return bool(text.strip()) and not native_text_usable(text, min_chars)


def _non_ascii_letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha() and ord(ch) > 127)


def _pixel_dark(pix) -> bool:
    return int(pix[0]) < 240 or int(pix[1]) < 240 or int(pix[2]) < 240


def ocr_text_usable(text: str, min_chars: int = NATIVE_TEXT_MIN_CHARS) -> bool:
    """Structural OCR filter only. Does not certify legal wording against the page image."""
    return native_text_usable(text, min_chars) and _non_ascii_letters(text) >= OCR_MIN_VIET_LETTERS


def _maxpool_dark(image, height: int, width: int, step_y: int, step_x: int) -> list[list[bool]]:
    """Downsample so a 1px stroke still marks its block. Blocks are OR of member pixels."""
    grid: list[list[bool]] = []
    for y0 in range(0, height, step_y):
        y1 = min(height, y0 + step_y)
        row: list[bool] = []
        for x0 in range(0, width, step_x):
            x1 = min(width, x0 + step_x)
            dark = False
            for y in range(y0, y1):
                pixrow = image[y]
                for x in range(x0, x1):
                    if _pixel_dark(pixrow[x]):
                        dark = True
                        break
                if dark:
                    break
            row.append(dark)
        grid.append(row)
    return grid


def _maxpool_dark_numpy(image, step_y: int, step_x: int):
    """Same max-pool as _maxpool_dark, on a raster already in memory as an array."""
    import numpy as np

    arr = np.asarray(image)
    dark = np.min(arr[:, :, :3], axis=2) < 240
    height, width = dark.shape
    pad_h = (step_y - height % step_y) % step_y
    pad_w = (step_x - width % step_x) % step_x
    if pad_h or pad_w:
        dark = np.pad(dark, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    cells_h = dark.shape[0] // step_y
    cells_w = dark.shape[1] // step_x
    return dark.reshape(cells_h, step_y, cells_w, step_x).max(axis=(1, 3))


def _ink_run_stats(grid) -> tuple[int, int, int]:
    sampled = 0
    ink = 0
    longest = 0
    if grid is None or len(grid) == 0:
        return 0, 0, 0
    for row in grid:
        run = 0
        for dark in row:
            sampled += 1
            if dark:
                ink += 1
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
    width = len(grid[0])
    for x in range(width):
        run = 0
        for row in grid:
            if row[x]:
                run += 1
                if run > longest:
                    longest = run
            else:
                run = 0
    return sampled, ink, longest


def page_looks_blank(image, ink_ratio_max: float = BLANK_INK_RATIO_MAX) -> bool:
    """True only for verified blank/noise. Thin strokes survive downsample via max-pool."""
    if isinstance(image, dict):
        return bool(image.get("blank", True))
    try:
        height, width, _ = image.shape
    except Exception:
        return False
    if height < 1 or width < 1:
        return True
    step_y = max(1, height // 128)
    step_x = max(1, width // 128)
    if getattr(image, "ndim", None) == 3:
        try:
            grid = _maxpool_dark_numpy(image, step_y, step_x)
        except Exception:
            grid = _maxpool_dark(image, height, width, step_y, step_x)
    else:
        grid = _maxpool_dark(image, height, width, step_y, step_x)
    sampled, ink, longest = _ink_run_stats(grid)
    if sampled == 0:
        return True
    if longest >= BLANK_INK_RUN_MIN:
        return False
    return (ink / sampled) < ink_ratio_max


def recovered_text_usable(recovered: str, native: str, min_chars: int = NATIVE_TEXT_MIN_CHARS) -> bool:
    """Accept OCR (or other recovery) only when it is usable and does not strip letters the native page already had."""
    if not native_text_usable(recovered, min_chars):
        return False
    native_vi = _non_ascii_letters(native)
    if native_vi >= RECOVERY_MIN_NATIVE_VIET and _non_ascii_letters(recovered) < native_vi * RECOVERY_KEEP_RATIO:
        return False
    return True


def read_bounded(chunks, max_bytes: int) -> bytes:
    """Stop a byte stream as soon as it passes the cap."""
    buf = bytearray()
    for chunk in chunks:
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise IndexFailure("pdf_oversized")
    return bytes(buf)


def extract_native(data: bytes, max_bytes: int) -> list[str]:
    return [text for _number, text in read_selected_pages(data, max_bytes, None, None)]


def read_selected_pages(data: bytes, max_bytes: int, max_pages: int | None, page_range: str | None) -> list[tuple[int, str]]:
    if len(data) > max_bytes:
        raise IndexFailure("pdf_oversized")
    if not data.startswith(b"%PDF-"):
        raise IndexFailure("mime_invalid")
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            raise IndexFailure("pdf_encrypted")
        count = len(reader.pages)
        start, end = page_bounds(page_range, count)
        if max_pages is not None and (end - start + 1) > max_pages:
            raise IndexFailure("page_limit")
        selected = []
        for number in range(start, end + 1):
            selected.append((number, normalize_text(reader.pages[number - 1].extract_text() or "")))
        return selected
    except IndexFailure:
        raise
    except Exception as exc:
        raise IndexFailure("pdf_corrupt") from exc
