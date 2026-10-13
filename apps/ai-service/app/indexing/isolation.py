"""Fail closed unless a P4B integration write targets the pinned test Compose project.

Hostname text is not enough. Development Postgres, MinIO, and Qdrant are also
reachable on localhost. A name that merely contains "test" is not enough either.
The accepted identity is the published host, port, database, bucket, Qdrant URL,
collection, and Compose project in deploy/docker-compose.p4b-test.yml.
"""

from __future__ import annotations

import os
import subprocess
from urllib.parse import unquote, urlparse

TEST_COMPOSE_PROJECT = "citizen-assistance-p4b-test"
TEST_DATABASE_NAME = "citizen_assistance_p4b_test"
TEST_BUCKET = "cas-p4b-integration-test"
TEST_POSTGRES_PORT = 5434
TEST_MINIO_ENDPOINTS = frozenset({"127.0.0.1:9010", "localhost:9010"})
TEST_QDRANT_URLS = frozenset({"http://127.0.0.1:6335", "http://localhost:6335"})
TEST_COLLECTIONS = frozenset({
    "p4b_test",
    "p4b_partial",
    "p4b_acknowledged",
    "p4b_dimension",
    "p4b_missing",
    "p4b_restart",
    "p4b_duplicate",
})
PUBLISHED_PORTS = (
    "127.0.0.1:5434->5432",
    "127.0.0.1:9010->9000",
    "127.0.0.1:9011->9001",
    "127.0.0.1:6335->6333",
)
ALLOWED_CONN_KEYS = frozenset({"user", "password", "dbname", "host", "port", "sslmode"})
LIBPQ_OVERRIDE_ENV = (
    "PGHOST",
    "PGHOSTADDR",
    "PGPORT",
    "PGDATABASE",
    "PGSERVICE",
    "PGSERVICEFILE",
    "PGSYSCONFDIR",
)


class IsolationError(RuntimeError):
    """The integration target is missing or is not the dedicated test stack."""


def _query_keys(url: str) -> list[str]:
    query = urlparse(url).query
    keys: list[str] = []
    for part in query.split("&"):
        if not part:
            continue
        keys.append(unquote(part.split("=", 1)[0]).strip().lower())
    return keys


def _require_libpq_target(url: str) -> None:
    """Accept only the pinned test server, using libpq's parameter rules.

    Query keys other than sslmode are rejected. libpq lets host, hostaddr,
    port, dbname, and service replace the values urlparse reads from the
    authority and path.
    """
    if not url.startswith(("postgres://", "postgresql://")):
        raise IsolationError("CAS_P4B_TEST_DATABASE_URL must be a postgres URL")
    keys = _query_keys(url)
    if len(keys) != len(set(keys)):
        raise IsolationError("duplicate PostgreSQL connection parameters")
    if any(key != "sslmode" for key in keys):
        raise IsolationError("PostgreSQL query parameter can change the connection target")
    try:
        import psycopg.conninfo as conninfo

        info = conninfo.conninfo_to_dict(url)
    except Exception as exc:
        raise IsolationError("PostgreSQL DSN is not accepted") from exc
    if set(info) - ALLOWED_CONN_KEYS:
        raise IsolationError("PostgreSQL DSN contains a parameter outside the test contract")
    host = str(info.get("host") or "")
    if not host or "," in host or host.startswith("/") or host.startswith("@"):
        raise IsolationError("refusing a multi-host or socket PostgreSQL target")
    try:
        port = int(str(info.get("port") or "5432"))
    except ValueError as exc:
        raise IsolationError("refusing a Postgres endpoint outside the dedicated test stack") from exc
    if (host, port) not in {("127.0.0.1", TEST_POSTGRES_PORT), ("localhost", TEST_POSTGRES_PORT)}:
        raise IsolationError("refusing a Postgres endpoint outside the dedicated test stack")
    if info.get("dbname") != TEST_DATABASE_NAME:
        raise IsolationError("refusing a database other than citizen_assistance_p4b_test")
    if info.get("sslmode", "disable") != "disable":
        raise IsolationError("the test database connection must disable TLS")
    if not info.get("user") or info.get("password") in (None, ""):
        raise IsolationError("test database credentials are required")


def _reject_libpq_env(env) -> None:
    """Refuse process environment that libpq can apply on top of a valid URL."""
    present = [name for name in LIBPQ_OVERRIDE_ENV if str(env.get(name, "")).strip()]
    if present:
        raise IsolationError("refusing libpq environment overrides: " + ",".join(present))


def lock_database_url(url: str) -> str:
    """Pin the test TCP address. This function is only for the isolated test DSN.

    A URL that already has a query uses `&`. A URL with no query uses `?`.
    Appending `&` to a URL without `?` makes libpq treat `hostaddr` as part of
    the database name, so the pin never applies. Production database URLs do
    not pass through this function.
    """
    _require_libpq_target(url)
    separator = "&" if urlparse(url).query else "?"
    return f"{url}{separator}hostaddr=127.0.0.1"


def assert_effective_endpoint(url: str) -> None:
    """Read the target libpq actually selected. Does not run a query."""
    from psycopg.pq import PGconn

    conn = PGconn.connect_start(url.encode())
    try:
        hostaddr = (conn.hostaddr or b"").decode()
        port = (conn.port or b"").decode()
        dbname = (conn.db or b"").decode()
    finally:
        conn.finish()
    if hostaddr != "127.0.0.1" or port != str(TEST_POSTGRES_PORT) or dbname != TEST_DATABASE_NAME:
        raise IsolationError("libpq resolved a PostgreSQL target outside the dedicated test stack")


def _qdrant_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise IsolationError("CAS_P4B_TEST_QDRANT_URL must be an http URL")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def isolated_p4b_targets(env: dict[str, str] | None = None) -> dict[str, str]:
    """Return the pinned test endpoints. Refuse anything that is not that stack."""
    source = os.environ if env is None else env
    project = source.get("CAS_P4B_TEST_COMPOSE_PROJECT", "").strip()
    database_url = source.get("CAS_P4B_TEST_DATABASE_URL", "").strip()
    bucket = source.get("CAS_P4B_TEST_BUCKET", "").strip()
    minio_endpoint = source.get("CAS_P4B_TEST_MINIO_ENDPOINT", "").strip()
    access_key = source.get("CAS_P4B_TEST_MINIO_ACCESS_KEY", "").strip()
    secret_key = source.get("CAS_P4B_TEST_MINIO_SECRET_KEY", "").strip()
    qdrant_url = source.get("CAS_P4B_TEST_QDRANT_URL", "").strip()
    if not all((project, database_url, bucket, minio_endpoint, access_key, secret_key, qdrant_url)):
        raise IsolationError("isolated P4B targets are required before any write")
    if project != TEST_COMPOSE_PROJECT:
        raise IsolationError("refusing a Compose project that is not citizen-assistance-p4b-test")
    _reject_libpq_env(source)
    if source is not os.environ:
        _reject_libpq_env(os.environ)
    database_url = lock_database_url(database_url)
    if bucket != TEST_BUCKET:
        raise IsolationError("refusing a bucket other than cas-p4b-integration-test")
    if minio_endpoint not in TEST_MINIO_ENDPOINTS:
        raise IsolationError("refusing a MinIO endpoint outside the dedicated test stack")
    normalized_qdrant = _qdrant_url(qdrant_url)
    if normalized_qdrant not in TEST_QDRANT_URLS:
        raise IsolationError("refusing a Qdrant endpoint outside the dedicated test stack")
    return {
        "compose_project": project,
        "database_url": database_url,
        "bucket": bucket,
        "minio_endpoint": minio_endpoint,
        "access_key": access_key,
        "secret_key": secret_key,
        "qdrant_url": normalized_qdrant,
    }


def require_test_collection(name: str) -> str:
    """Refuse the development collection and any name the test file does not use."""
    if name not in TEST_COLLECTIONS:
        raise IsolationError("refusing a Qdrant collection outside the dedicated test stack")
    return name


def require_compose_ports(listing: str) -> None:
    """The listing must be `docker ps` output for the test project only."""
    if TEST_COMPOSE_PROJECT not in listing:
        raise IsolationError("docker did not show the dedicated test Compose project")
    for token in PUBLISHED_PORTS:
        if token not in listing:
            raise IsolationError("dedicated test ports are not published by the test Compose project")
    if "0.0.0.0:" in listing or "[::]:" in listing:
        raise IsolationError("test project publishes a port outside loopback")
    for forbidden in ("127.0.0.1:5432->5432", "127.0.0.1:9000->9000", "127.0.0.1:6333->6333"):
        if forbidden in listing:
            raise IsolationError("test project listing publishes a development port")


def read_compose_ps() -> str:
    """Ports published by the pinned project. Empty or failed docker output is a refusal."""
    try:
        completed = subprocess.run(
            [
                "docker", "ps",
                "--filter", f"label=com.docker.compose.project={TEST_COMPOSE_PROJECT}",
                "--format", "{{.Names}} {{.Ports}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise IsolationError("cannot inspect the dedicated test Compose project") from exc
    if completed.returncode != 0:
        raise IsolationError("cannot inspect the dedicated test Compose project")
    listing = completed.stdout or ""
    require_compose_ports(listing)
    return listing


def connect_test_database(database_url: str):
    """Open the same locked target that isolation already verified."""
    _reject_libpq_env(os.environ)
    assert_effective_endpoint(database_url)
    import psycopg

    connection = psycopg.connect(database_url)
    info = connection.info
    if info.hostaddr != "127.0.0.1" or str(info.port) != str(TEST_POSTGRES_PORT):
        connection.close()
        raise IsolationError("connected PostgreSQL server is outside the dedicated test stack")
    return connection


def read_current_database(database_url: str) -> str:
    with connect_test_database(database_url) as conn:
        row = conn.execute("SELECT current_database()").fetchone()
    if row is None or row[0] != TEST_DATABASE_NAME:
        raise IsolationError("connected database is not citizen_assistance_p4b_test")
    return row[0]


def prepare_isolated_target(env: dict[str, str] | None = None, *, inspect=None) -> dict[str, str]:
    """Validate the static identity first. The inspect hook runs only after that.

    Integration tests omit `inspect` and then prove the Compose project publishes
    the test ports and PostgreSQL reports the test database name. A failed static
    check never calls inspect and therefore never opens a connection.
    """
    target = isolated_p4b_targets(env)
    check = inspect if inspect is not None else _live_inspect
    check(target)
    return target


def _live_inspect(target: dict[str, str]) -> None:
    read_compose_ps()
    read_current_database(target["database_url"])
