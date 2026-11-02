"""Read objects that the API's file store wrote (OBJECT_STORAGE_BACKEND=file).

An object lives at <root>/<bucket>/<key>; the key contract is
deploy/contracts/object_store_keys.json, shared with apps/api/internal/storage/file.go.
Every path component is opened relative to its parent with O_NOFOLLOW, starting
from a descriptor of the root, so neither a ".." key nor a symlink anywhere below
the root can make this reader open a file outside it. The ai-service only reads;
the API writes (atomically: temporary file, fsync, rename).
"""

from __future__ import annotations

import errno
import os
import re
import stat

from app.config import ConfigError
from app.indexing.errors import IndexFailure

MAX_KEY_BYTES = 512
MAX_KEY_COMPONENTS = 8
_COMPONENT = re.compile(r"[A-Za-z0-9._-]{1,128}")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
# O_NONBLOCK: opening a FIFO planted at an object path must not block; blocking mode is
# restored once the descriptor is known to be a regular file.
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
_UNSAFE = {errno.ELOOP, errno.ENOTDIR, errno.EMLINK}


def backend() -> str:
    value = os.getenv("OBJECT_STORAGE_BACKEND", "s3").strip().lower() or "s3"
    if value not in ("s3", "file"):
        raise ConfigError("OBJECT_STORAGE_BACKEND must be s3 or file")
    return value


def file_root() -> str:
    root = os.getenv("OBJECT_STORAGE_FILE_ROOT", "").strip()
    if not root or not os.path.isabs(root) or os.path.normpath(root) != root:
        raise ConfigError("OBJECT_STORAGE_FILE_ROOT must be an absolute clean path when OBJECT_STORAGE_BACKEND=file")
    return root


def valid_component(component: str) -> bool:
    return bool(_COMPONENT.fullmatch(component)) and component[0] not in ".-"


def valid_key(key: str) -> bool:
    if not isinstance(key, str) or not key or len(key.encode("utf-8")) > MAX_KEY_BYTES:
        return False
    parts = key.split("/")
    return len(parts) <= MAX_KEY_COMPONENTS and all(valid_component(part) for part in parts)


def _open_dirs(root: str, components: list[str]) -> int:
    fd = os.open(root, _DIR_FLAGS)
    try:
        for component in components:
            nxt = os.open(component, _DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def bucket_ready(root: str, bucket: str) -> bool:
    """True when <root>/<bucket> is a real, readable directory (no symlink on the way)."""
    if not valid_component(bucket):
        return False
    try:
        fd = _open_dirs(root, [bucket])
    except OSError:
        return False
    os.close(fd)
    return True


def read_object(root: str, bucket: str, key: str, max_bytes: int, deadline=None) -> bytes:
    """Read one object, refusing traversal, symlinks, non-regular files and oversize."""
    if not valid_component(bucket) or not valid_key(key):
        raise IndexFailure("source_unsafe_path")
    parts = key.split("/")
    try:
        dfd = _open_dirs(root, [bucket, *parts[:-1]])
    except FileNotFoundError as exc:
        raise IndexFailure("source_not_found") from exc
    except OSError as exc:
        if exc.errno in _UNSAFE:
            raise IndexFailure("source_unsafe_path") from exc
        raise IndexFailure("source_timeout") from exc
    try:
        try:
            fd = os.open(parts[-1], _FILE_FLAGS, dir_fd=dfd)
        except FileNotFoundError as exc:
            raise IndexFailure("source_not_found") from exc
        except OSError as exc:
            if exc.errno in _UNSAFE:
                raise IndexFailure("source_unsafe_path") from exc
            raise IndexFailure("source_timeout") from exc
    finally:
        os.close(dfd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise IndexFailure("source_unsafe_path")
        os.set_blocking(fd, True)
        if info.st_size > max_bytes:
            raise IndexFailure("pdf_oversized")
        chunks = bytearray()
        while True:
            if deadline is not None:
                deadline.check()
            block = os.read(fd, 64 * 1024)
            if not block:
                break
            chunks.extend(block)
            if len(chunks) > max_bytes:
                raise IndexFailure("pdf_oversized")
        return bytes(chunks)
    finally:
        os.close(fd)
