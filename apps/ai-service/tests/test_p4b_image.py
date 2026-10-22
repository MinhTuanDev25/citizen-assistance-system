from pathlib import Path


def test_production_image_pins_cpu_stack():
    root = Path(__file__).resolve().parents[1]
    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    prod = (root / "requirements-prod.txt").read_text(encoding="utf-8")
    assert "python:3.11-slim-bookworm" in dockerfile
    assert "requirements-prod.txt" in dockerfile
    for pin in (
        "onnxruntime==1.20.1",
        "tokenizers==0.20.3",
        "numpy==1.26.4",
        "paddlepaddle==3.0.0",
        "paddleocr==2.9.1",
        "pypdfium2==4.30.0",
    ):
        assert pin in prod
    dev = (root / "requirements.txt").read_text(encoding="utf-8")
    assert "paddleocr" not in dev
    assert "onnxruntime" not in dev
