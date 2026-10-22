#!/usr/bin/env python3
"""Create the indexing collection once. Does not run from /ready."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "apps", "ai-service"))

from app.indexing.runtime import provision_qdrant_collection  # noqa: E402


def main() -> int:
    url = os.environ.get("QDRANT_URL", "").strip()
    collection = os.environ.get("QDRANT_COLLECTION", "knowledge_chunks").strip()
    if not url or not collection:
        print("QDRANT_URL and QDRANT_COLLECTION are required", file=sys.stderr)
        return 2
    provision_qdrant_collection(url, collection)
    print("qdrant_collection_ready", collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
