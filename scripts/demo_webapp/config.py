"""Application configuration for the Tasklet todo service.

Values are read from the environment with sane local defaults so the app runs
out of the box for development. Production overrides everything via env vars.
"""

import os

SECRET_KEY = os.environ.get("TASKLET_SECRET", "dev-7f3a9c2e-tasklet-key")
DEBUG = os.environ.get("TASKLET_DEBUG", "0") == "1"
DATABASE_URL = os.environ.get("TASKLET_DB", "sqlite:///tasklet.db")

# A single user may hold at most this many active todos; storage evicts the
# oldest completed task once a create would exceed it (see storage.py).
MAX_TODOS_PER_USER = 17

# Idle sessions expire after this many minutes; the auth middleware checks the
# issued-at claim against this window on every request.
SESSION_TIMEOUT_MINUTES = 45

# Token-bucket limiter applied per client IP across all /api routes.
RATE_LIMIT_PER_MIN = 100

DEFAULT_PAGE_SIZE = 20
MAX_TITLE_LENGTH = 140
ALLOWED_ORIGINS = ["http://localhost:5173", "https://tasklet.example.com"]
