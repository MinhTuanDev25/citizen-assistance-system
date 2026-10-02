#!/usr/bin/env python3
"""Run one ONNX forward pass and one PaddleOCR page. Never downloads models.

Exit 0 and print P4B_DONE only when both real providers pass.
Exit 2 and print P4B_MODEL_SMOKE_PENDING when the local model directories are absent.
The output contains counts and codes only, never extracted text.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    embed_dir = os.getenv("EMBEDDING_MODEL_DIR", "").strip()
    ocr_dir = os.getenv("OCR_MODEL_DIR", "").strip()
    if not embed_dir or not ocr_dir or not Path(embed_dir, "manifest.json").is_file() or not Path(ocr_dir, "manifest.json").is_file():
        print("manifest_verification=pending")
        print("runtime_compatibility=pending")
        print("forward_pass=pending")
        print("P4B_MODEL_SMOKE_PENDING")
        return 2
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "apps" / "ai-service"))
    from app.indexing.embed import OnnxEmbedder
    from app.indexing.ocr import PaddleOCR, PdfiumRenderer

    embedder = OnnxEmbedder(embed_dir)
    vector = embedder.embed("passage check")
    if len(vector) != 384 or any(item != item or abs(item) == float("inf") for item in vector):
        print("P4B_MODEL_SMOKE_FAIL embedding")
        return 1
    norm = sum(item * item for item in vector) ** 0.5
    if abs(norm - 1.0) > 1e-3:
        print("P4B_MODEL_SMOKE_FAIL norm")
        return 1
    fixture = root / "apps" / "ai-service" / "tests" / "fixtures" / "mixed.pdf"
    data = fixture.read_bytes()
    image = PdfiumRenderer().render(data, 2, 150)
    text = PaddleOCR(ocr_dir).read_page(image, 2)
    if not text:
        print("P4B_MODEL_SMOKE_FAIL ocr_empty")
        return 1
    if embedder.revision != "6a0d452a575215f80b8f66276dd4ee5d504942c6" or embedder.checksum != "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88":
        print("P4B_MODEL_SMOKE_FAIL manifest")
        return 1
    print(f"P4B_DONE embedding_dim={len(vector)} ocr_chars={len(text)} image_shape={getattr(image, 'shape', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
