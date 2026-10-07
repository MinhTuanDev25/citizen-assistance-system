"""Embedding providers. Document chunks use the passage prefix. Query prefix is reserved."""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import threading
import time
from pathlib import Path

from app.indexing.errors import IndexFailure

E5_MODEL_ID = "intfloat/multilingual-e5-small"
E5_SOURCE_REPOSITORY = "https://huggingface.co/intfloat/multilingual-e5-small"
# Commit that added onnx/model_qint8_avx512_vnni.onnx. Resolved from the
# Hugging Face commits API on 2026-10-01. The previous pin 6e0d5e48… is not a commit.
E5_REVISION = "6a0d452a575215f80b8f66276dd4ee5d504942c6"
E5_SOURCE_FILE = "onnx/model_qint8_avx512_vnni.onnx"
E5_TOKENIZER_SOURCE_FILE = "onnx/tokenizer.json"
E5_DIMENSION = 384
E5_MAX_TOKENS = 512
E5_QUANTIZATION = "int8"
E5_QUANTIZATION_METHOD = "qint8_avx512_vnni"
# LFS oids published at E5_REVISION. embedding_checksum is the model file digest.
E5_MODEL_SHA256 = "dd476dd0c2514e9b9be83aeb3853fac0763e0bdf4a71645407587d77c48a2d88"
E5_TOKENIZER_SHA256 = "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39"
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
FAKE_MODEL_ID = "fake-embedding"
FAKE_REVISION = "deterministic-v1"


# The pinned qint8_avx512_vnni graph uses these AVX-512 features.
# Linux /proc/cpuinfo spells VNNI as avx512_vnni. Other sources use avx512vnni.
REQUIRED_CPU_FLAGS = ("avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512vnni")
_VNNI_ALIASES = {"avx512_vnni": "avx512vnni", "avx512vnni": "avx512vnni"}


def normalize_cpu_flags(items) -> set[str]:
    out: set[str] = set()
    for item in items:
        token = str(item).strip().lower()
        if not token:
            continue
        out.add(_VNNI_ALIASES.get(token, token))
    return out


def parse_proc_cpuinfo(text: str) -> set[str]:
    """Parse Linux /proc/cpuinfo. The flags line uses avx512_vnni, not avx512vnni."""
    flags: set[str] = set()
    for line in text.splitlines():
        lowered = line.lower()
        if lowered.startswith("flags") or lowered.startswith("features"):
            flags.update(normalize_cpu_flags(line.split(":", 1)[1].split()))
            break
    return flags


def parse_sysctl(text: str) -> set[str]:
    flags: set[str] = set()
    for line in text.splitlines():
        if ": 1" not in line:
            continue
        name = line.split(":", 1)[0].strip().lower()
        leaf = name.rsplit(".", 1)[-1]
        flags.update(normalize_cpu_flags([leaf]))
    return flags


def detect_cpu_flags(cpuinfo_text: str | None = None, sysctl_text: str | None = None) -> set[str]:
    """Read the OS CPU feature list. EMBEDDING_CPU_FLAGS is an explicit override only.

    Passing cpuinfo_text or sysctl_text uses that text and ignores the override,
    so a test can exercise the Linux parser without hiding it behind the env var.
    """
    if cpuinfo_text is None and sysctl_text is None and "EMBEDDING_CPU_FLAGS" in os.environ:
        raw = os.environ.get("EMBEDDING_CPU_FLAGS") or ""
        return normalize_cpu_flags(raw.replace(",", " ").split())
    flags: set[str] = set()
    if cpuinfo_text is None and sysctl_text is None:
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            cpuinfo_text = cpuinfo.read_text(encoding="utf-8", errors="replace")
    if cpuinfo_text:
        flags.update(parse_proc_cpuinfo(cpuinfo_text))
    if sysctl_text is None and cpuinfo_text is None and platform.system() == "Darwin":
        try:
            sysctl_text = subprocess.check_output(
                ["sysctl", "-a"], text=True, timeout=2, stderr=subprocess.DEVNULL
            )
        except (OSError, subprocess.SubprocessError):
            sysctl_text = ""
    if sysctl_text:
        flags.update(parse_sysctl(sysctl_text))
    return flags


def require_supported_variant(flags: set[str] | None = None) -> str:
    """Select the pinned qint8 graph only when every required AVX-512 flag is present.

    No fallback filename is invented. An unsupported CPU fails before ONNX load.
    """
    seen = detect_cpu_flags() if flags is None else normalize_cpu_flags(flags)
    if any(name not in seen for name in REQUIRED_CPU_FLAGS):
        raise IndexFailure("embedding_cpu_unsupported")
    return E5_QUANTIZATION_METHOD


def fake_embed(text: str, dimension: int = E5_DIMENSION) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    counter = 0
    while len(values) < dimension:
        block = hashlib.sha256(digest + counter.to_bytes(4, "big")).digest()
        counter += 1
        for i in range(0, len(block), 4):
            raw = int.from_bytes(block[i : i + 4], "big")
            values.append((raw / 4294967295) * 2 - 1)
            if len(values) == dimension:
                break
    return normalize(values)


def normalize(values: list[float]) -> list[float]:
    if any(not math.isfinite(item) for item in values):
        raise IndexFailure("embedding_non_finite")
    norm = math.sqrt(sum(item * item for item in values))
    if norm == 0 or not math.isfinite(norm):
        raise IndexFailure("embedding_non_finite")
    return [item / norm for item in values]


def fake_checksum() -> str:
    return hashlib.sha256(b"fake-embedding-deterministic-v1").hexdigest()


def load_manifest(model_dir: str) -> dict:
    path = Path(model_dir)
    manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        raise IndexFailure("embedding_model_missing")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IndexFailure("embedding_checksum_mismatch") from exc
    if manifest.get("model_id") != E5_MODEL_ID or manifest.get("revision") != E5_REVISION:
        raise IndexFailure("embedding_checksum_mismatch")
    if manifest.get("source_repository") != E5_SOURCE_REPOSITORY:
        raise IndexFailure("embedding_checksum_mismatch")
    if manifest.get("source_file") != E5_SOURCE_FILE:
        raise IndexFailure("embedding_checksum_mismatch")
    if manifest.get("dimension") != E5_DIMENSION:
        raise IndexFailure("embedding_dimension")
    if manifest.get("quantization") != E5_QUANTIZATION or manifest.get("quantization_method") != E5_QUANTIZATION_METHOD:
        raise IndexFailure("embedding_checksum_mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise IndexFailure("embedding_model_missing")
    expected_files = {"model.onnx": E5_MODEL_SHA256, "tokenizer.json": E5_TOKENIZER_SHA256}
    if set(files) != set(expected_files) or any(files[name] != digest for name, digest in expected_files.items()):
        raise IndexFailure("embedding_checksum_mismatch")
    for name, expected in expected_files.items():
        file_path = path / name
        if not file_path.is_file():
            raise IndexFailure("embedding_model_missing")
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != expected:
            raise IndexFailure("embedding_checksum_mismatch")
    return manifest


class OnnxEmbedder:
    """CPU Int8 ONNX adapter. Tokenizer is loaded once. This class never downloads."""

    def __init__(self, model_dir: str, batch_size: int = 8) -> None:
        self.manifest = load_manifest(model_dir)
        self.dimension = E5_DIMENSION
        self.model_id = E5_MODEL_ID
        self.revision = E5_REVISION
        self.max_tokens = E5_MAX_TOKENS
        self.batch_size = max(1, batch_size)
        # SHA-256 of model.onnx, the upstream LFS oid of the pinned source file.
        self.checksum = E5_MODEL_SHA256
        self.variant = require_supported_variant()
        self.cpu_compatible = True
        self._dir = Path(model_dir)
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise IndexFailure("embedding_model_missing") from exc
        self._tokenizer = Tokenizer.from_file(str(self._dir / "tokenizer.json"))
        self._tokenizer.no_truncation()
        self._session = onnxruntime.InferenceSession(
            str(self._dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [item.name for item in self._session.get_inputs()]

    def count_tokens(self, text: str, deadline=None) -> int:
        if deadline is not None:
            deadline.check()
        encoded = self._tokenizer.encode(PASSAGE_PREFIX + text, add_special_tokens=True)
        if deadline is not None:
            deadline.check()
        return len(encoded.ids)

    def split_text(self, text: str, limit: int, deadline=None) -> list[str]:
        return split_to_limit(self, text, limit, deadline)

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str], deadline=None) -> list[list[float]]:
        if deadline is not None:
            deadline.check()
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            out.extend(self._run(texts[start : start + self.batch_size]))
        return out

    def _run(self, texts: list[str]) -> list[list[float]]:
        import numpy as np

        for text in texts:
            if self.count_tokens(text) > E5_MAX_TOKENS:
                raise IndexFailure("chunk_limit")
        encoded = [self._tokenizer.encode(PASSAGE_PREFIX + text, add_special_tokens=True) for text in texts]
        width = max(len(item.ids) for item in encoded)
        ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        types = np.zeros((len(encoded), width), dtype=np.int64)
        for row, item in enumerate(encoded):
            ids[row, : len(item.ids)] = item.ids
            mask[row, : len(item.ids)] = item.attention_mask or [1] * len(item.ids)
            type_ids = getattr(item, "type_ids", None) or [0] * len(item.ids)
            types[row, : len(type_ids)] = type_ids
        feeds = {}
        for name in self._input_names:
            if name == "input_ids":
                feeds[name] = ids
            elif name == "attention_mask":
                feeds[name] = mask
            elif name == "token_type_ids":
                feeds[name] = types
            else:
                raise IndexFailure("embedding_model_missing")
        hidden = self._session.run(None, feeds)[0]
        if getattr(hidden, "ndim", 1) == 3:
            weights = mask.astype(np.float32)
            pooled = (hidden * weights[:, :, None]).sum(axis=1) / np.maximum(weights.sum(axis=1, keepdims=True), 1.0)
        elif getattr(hidden, "ndim", 1) == 2:
            pooled = hidden
        else:
            raise IndexFailure("embedding_dimension")
        vectors = []
        for row in pooled:
            values = [float(item) for item in row.reshape(-1).tolist()]
            if len(values) != self.dimension:
                raise IndexFailure("embedding_dimension")
            vectors.append(normalize(values))
        return vectors


def split_to_limit(counter, text: str, limit: int, deadline=None) -> list[str]:
    """Split on character boundaries using the counter. Concatenation equals text."""

    def tokens(value: str) -> int:
        if deadline is None:
            return counter.count_tokens(value)
        return counter.count_tokens(value, deadline=deadline)

    if tokens(text) <= limit:
        return [text]
    pieces: list[str] = []
    rest = text
    while rest:
        if tokens(rest) <= limit:
            pieces.append(rest)
            break
        lo, hi, best = 1, len(rest) - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if tokens(rest[:mid]) <= limit:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        if best < 1:
            raise IndexFailure("chunk_limit")
        pieces.append(rest[:best])
        rest = rest[best:]
    return pieces


class SubwordCounter:
    """Subword double. Truncation is opt-in so tests can show the old bug."""

    def __init__(self, width: int = 1, overhead: int = 4, limit: int = E5_MAX_TOKENS, truncate: bool = False) -> None:
        self.width = width
        self.overhead = overhead
        self.limit = limit
        self.truncate = truncate
        self.max_tokens = limit
        self.model_id = FAKE_MODEL_ID
        self.revision = FAKE_REVISION
        self.checksum = fake_checksum()
        self.dimension = E5_DIMENSION

    def count_tokens(self, text: str, deadline=None) -> int:
        if deadline is not None:
            deadline.check()
        content = (len(text) + self.width - 1) // self.width if text else 0
        total = self.overhead + content
        if self.truncate:
            return min(total, self.limit)
        return total

    def split_text(self, text: str, limit: int, deadline=None) -> list[str]:
        return split_to_limit(self, text, limit, deadline)

    def embed_batch(self, texts: list[str], deadline=None) -> list[list[float]]:
        if deadline is not None:
            deadline.check()
        vectors = []
        for text in texts:
            if self.count_tokens(text, deadline=deadline) > self.limit:
                raise IndexFailure("chunk_limit")
            vectors.append(fake_embed(PASSAGE_PREFIX + text, self.dimension))
        return vectors


class WordTokenCounter:
    """Initial split helper. The embedding tokenizer is the hard-limit authority."""

    def count_tokens(self, text: str, deadline=None) -> int:
        if deadline is not None:
            deadline.check()
        from app.indexing.chunk import tokenize

        return len(tokenize(PASSAGE_PREFIX + text)) + 2


def _onnx_worker(conn, model_dir: str, batch_size: int, behavior: str) -> None:
    from app.indexing.procgroup import arm_parent_death

    arm_parent_death()
    if behavior == "init_hang":
        time_sleep(3600)
    if behavior == "init_crash":
        os._exit(2)
    embedder = None
    variant = E5_QUANTIZATION_METHOD
    if behavior == "onnx":
        embedder = OnnxEmbedder(model_dir, batch_size=batch_size)
        variant = embedder.variant
    if behavior == "broken":
        conn.send({"event": "ready", "variant": variant})
        os._exit(5)
    conn.send({"event": "ready", "variant": variant})
    while True:
        try:
            msg = conn.recv()
        except EOFError:
            return
        if not isinstance(msg, dict) or msg.get("op") == "shutdown":
            return
        if behavior == "hang":
            time_sleep(3600)
        if behavior == "crash":
            os._exit(3)
        if behavior == "eof":
            conn.close()
            return
        if behavior == "malformed":
            conn.send("bad")
            continue
        request_id = msg.get("id")
        if behavior != "onnx":
            time_sleep(0.02)
            if msg.get("op") == "count":
                conn.send({"event": "result", "id": request_id, "n": len(msg.get("text") or "")})
            elif msg.get("op") == "embed":
                texts = list(msg.get("texts") or [])
                conn.send({"event": "result", "id": request_id, "vectors": [[0.1] * E5_DIMENSION for _ in texts]})
            else:
                conn.send({"event": "failed", "id": request_id})
            continue
        if msg.get("op") == "count":
            conn.send({"event": "result", "id": request_id, "n": embedder.count_tokens(msg.get("text") or "")})
        elif msg.get("op") == "embed":
            conn.send({"event": "result", "id": request_id, "vectors": embedder.embed_batch(list(msg.get("texts") or []))})
        else:
            conn.send({"event": "failed", "id": request_id})


def time_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


class OnnxProcess:
    """One ONNX child. Pipe traffic is serialized so concurrent callers cannot swap replies."""

    def __init__(self, model_dir: str, batch_size: int = 8, behavior: str = "onnx") -> None:
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.behavior = behavior
        self.recover_behavior: str | None = None
        self.max_respawns = 1
        self.model_id = E5_MODEL_ID
        self.revision = E5_REVISION
        self.checksum = E5_MODEL_SHA256
        self.dimension = E5_DIMENSION
        self.max_tokens = E5_MAX_TOKENS
        self.variant = None
        self.cpu_compatible = False
        self.status = "initializing"
        self.spawn_count = 0
        self.reaped = 0
        self._consecutive_failures = 0
        self._seq = 0
        self._init_timeout = 30.0
        self._ctx = mp.get_context("spawn")
        self._proc = None
        self._conn = None
        self._lock = threading.Lock()

    def start(self, timeout_s: float) -> None:
        self._init_timeout = timeout_s
        self._boot(self.behavior, timeout_s)

    def is_alive(self) -> bool:
        proc = self._proc
        return self.status == "ready" and proc is not None and proc.is_alive()

    def count_tokens(self, text: str, deadline=None, timeout_s: float = 30) -> int:
        msg = self._call({"op": "count", "text": text}, timeout_s, deadline)
        return int(msg["n"])

    def split_text(self, text: str, limit: int, deadline=None) -> list[str]:
        return split_to_limit(self, text, limit, deadline)

    def embed_batch(self, texts: list[str], deadline=None, timeout_s: float = 60) -> list[list[float]]:
        msg = self._call({"op": "embed", "texts": texts}, timeout_s, deadline)
        return msg["vectors"]

    def shutdown(self) -> None:
        """Kill the child without waiting for the pipe lock held by an in-flight call."""
        self._disable("stopped")

    def _boot(self, behavior: str, timeout_s: float) -> None:
        self.status = "initializing"
        if behavior == "onnx":
            self.variant = require_supported_variant()
            self.cpu_compatible = True
        self._disable("initializing")
        parent, child = self._ctx.Pipe(duplex=True)
        proc = self._ctx.Process(
            target=_onnx_worker,
            args=(child, self.model_dir, self.batch_size, behavior),
            daemon=False,
        )
        self.spawn_count += 1
        proc.start()
        child.close()
        self._proc = proc
        self._conn = parent
        if not parent.poll(timeout_s):
            self._disable("timed_out")
            raise IndexFailure("embedding_model_missing")
        try:
            msg = parent.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._disable("failed")
            raise IndexFailure("embedding_model_missing") from exc
        if not isinstance(msg, dict) or msg.get("event") != "ready":
            self._disable("failed")
            raise IndexFailure("embedding_model_missing")
        self.variant = msg.get("variant") or self.variant
        self.cpu_compatible = True
        self.status = "ready"

    def _budget(self, timeout_s: float, deadline) -> float:
        if deadline is None:
            return timeout_s
        return deadline.timeout_for_io(timeout_s)

    def _call(self, payload: dict, timeout_s: float, deadline=None) -> dict:
        wait = self._budget(timeout_s, deadline)
        acquired = self._acquire(wait, deadline)
        if not acquired:
            raise IndexFailure("timeout")
        try:
            self._seq += 1
            message = dict(payload)
            message["id"] = self._seq
            try:
                msg = self._exchange(message, self._budget(timeout_s, deadline), deadline)
            except IndexFailure:
                if not self._can_retry(deadline, timeout_s):
                    raise
                msg = self._exchange(message, self._budget(timeout_s, deadline), deadline)
            self._consecutive_failures = 0
            return msg
        finally:
            self._lock.release()

    def _acquire(self, wait: float, deadline) -> bool:
        end = time.monotonic() + wait
        while True:
            if deadline is not None:
                deadline.check()
            if self._lock.acquire(timeout=min(0.02, max(0.0, end - time.monotonic()))):
                return True
            if time.monotonic() >= end:
                return False

    def _can_retry(self, deadline, timeout_s: float) -> bool:
        if self.status == "stopped" or self.max_respawns <= 0 or self._consecutive_failures >= self.max_respawns:
            self.status = "failed" if self.status == "ready" else self.status
            return False
        try:
            left = self._budget(timeout_s, deadline)
        except IndexFailure:
            return False
        if left <= 0:
            return False
        self._consecutive_failures += 1
        return self._recover(left)

    def _wait_result(self, timeout_s: float, deadline=None):
        """Poll in short slices so shutdown or a cancelled deadline stops the call."""
        end = time.monotonic() + timeout_s
        while True:
            if deadline is not None:
                try:
                    deadline.check()
                except IndexFailure:
                    self._disable("timed_out")
                    raise
            conn = self._conn
            if conn is None or self.status == "stopped":
                raise IndexFailure("timeout")
            left = end - time.monotonic()
            if left <= 0:
                self._disable("timed_out")
                raise IndexFailure("timeout")
            try:
                ready = conn.poll(min(0.05, left))
            except (EOFError, OSError) as exc:
                self._disable("failed")
                raise IndexFailure("embedding_model_missing") from exc
            if ready:
                return conn

    def _exchange(self, payload: dict, timeout_s: float, deadline=None) -> dict:
        if self.status != "ready" or self._conn is None:
            raise IndexFailure("embedding_model_missing")
        if deadline is not None:
            deadline.check()
        try:
            self._conn.send(payload)
            conn = self._wait_result(timeout_s, deadline)
            msg = conn.recv()
        except IndexFailure:
            raise
        except (EOFError, BrokenPipeError, OSError) as exc:
            self._disable("failed")
            raise IndexFailure("embedding_model_missing") from exc
        if not isinstance(msg, dict) or msg.get("event") != "result" or msg.get("id") != payload.get("id"):
            self._disable("failed")
            raise IndexFailure("embedding_model_missing")
        return msg

    def _recover(self, timeout_s: float) -> bool:
        if timeout_s <= 0:
            self.status = "timed_out"
            return False
        behavior = self.recover_behavior or self.behavior
        try:
            self._boot(behavior, timeout_s)
        except IndexFailure:
            if self.status not in ("timed_out", "failed"):
                self.status = "failed"
            return False
        return self.status == "ready"

    def _disable(self, status: str) -> None:
        conn, proc = self._conn, self._proc
        self._conn = None
        self._proc = None
        self._reap(proc, conn)
        self.status = status

    def _reap(self, proc, conn) -> None:
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        if proc is None:
            return
        if proc.is_alive():
            proc.kill()
        proc.join(timeout=5)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5)
        self.reaped += 1
