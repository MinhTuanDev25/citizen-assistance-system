"""Rate limit runs after auth and before the provider. No live network."""

import threading

from fastapi.testclient import TestClient

import app.main as main
from app.main import app
from app.ratelimit import ExtractLimiter
from tests.conftest import AUTH_HEADER, TEST_SERVICE_TOKEN
from tests.test_extract_endpoint import VALID_BODY

client = TestClient(app)


def test_limiter_rate_then_concurrency():
    limiter = ExtractLimiter(rate_per_minute=2, max_inflight=1, retry_after_s=7)
    ok, reason = limiter.try_acquire()
    assert ok and reason == ""
    ok, reason = limiter.try_acquire()
    assert ok is False and reason == "concurrency_limited"
    limiter.release()
    ok, _ = limiter.try_acquire()
    assert ok is True
    ok, reason = limiter.try_acquire()
    assert ok is False and reason == "rate_limited"
    limiter.release()


def test_extract_rate_limit_returns_429_with_retry_after(monkeypatch):
    calls = {"n": 0}
    real = main.get_extract_service().extract

    def counting(request):
        calls["n"] += 1
        return real(request)

    monkeypatch.setattr(main.get_extract_service(), "extract", counting)
    monkeypatch.setattr(main, "_limiter", ExtractLimiter(1, 4, 9))
    first = client.post("/v1/extract", json=VALID_BODY, headers=AUTH_HEADER)
    second = client.post("/v1/extract", json=VALID_BODY, headers=AUTH_HEADER)
    assert first.status_code == 200
    assert second.status_code == 429
    assert second.headers["retry-after"] == "9"
    assert second.json()["detail"]["reason"] == "rate_limited"
    assert calls["n"] == 1
    assert TEST_SERVICE_TOKEN not in second.text


def test_extract_concurrency_limit_does_not_call_provider(monkeypatch):
    calls = {"n": 0}

    def explode(request):
        calls["n"] += 1
        raise AssertionError("provider must not run")

    monkeypatch.setattr(main.get_extract_service(), "extract", explode)
    monkeypatch.setattr(main, "_limiter", ExtractLimiter(10, 1, 4))
    # Hold the only inflight slot, then the HTTP call must be rejected first.
    assert main.get_limiter().try_acquire()[0] is True
    response = client.post("/v1/extract", json=VALID_BODY, headers=AUTH_HEADER)
    main.get_limiter().release()
    assert response.status_code == 429
    assert response.headers["retry-after"] == "4"
    assert response.json()["detail"]["reason"] == "concurrency_limited"
    assert calls["n"] == 0


def test_unauthorized_does_not_consume_rate_limit(monkeypatch):
    monkeypatch.setattr(main, "_limiter", ExtractLimiter(1, 1, 3))
    denied = client.post("/v1/extract", json=VALID_BODY)
    allowed = client.post("/v1/extract", json=VALID_BODY, headers=AUTH_HEADER)
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert TEST_SERVICE_TOKEN not in denied.text


def test_limiter_threads_do_not_exceed_inflight():
    limiter = ExtractLimiter(rate_per_minute=100, max_inflight=1, retry_after_s=1)
    started = threading.Event()
    release = threading.Event()
    results = []

    def hold():
        ok, _ = limiter.try_acquire()
        results.append(("hold", ok))
        started.set()
        release.wait(timeout=2)
        limiter.release()

    def second():
        started.wait(timeout=2)
        results.append(("second", limiter.try_acquire()))

    t1 = threading.Thread(target=hold)
    t2 = threading.Thread(target=second)
    t1.start()
    t2.start()
    t2.join(timeout=2)
    release.set()
    t1.join(timeout=2)
    assert ("hold", True) in results
    assert ("second", (False, "concurrency_limited")) in results
