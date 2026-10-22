"""Set a service token before any test imports the FastAPI app.

pytest loads this file before test modules. load_settings() runs at import
of app.main and refuses to start without AI_SERVICE_TOKEN.
"""

import os

os.environ.setdefault("AI_SERVICE_TOKEN", "test-service-token")
os.environ.setdefault("LLM_PROVIDER", "mock")

TEST_SERVICE_TOKEN = os.environ["AI_SERVICE_TOKEN"]
AUTH_HEADER = {"Authorization": f"Bearer {TEST_SERVICE_TOKEN}"}
