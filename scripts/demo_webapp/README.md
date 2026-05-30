# Tasklet

A minimal todo service: a Flask JSON API over an in-memory store, with per-user
task quotas and a token-bucket rate limiter. Built as a small, readable example.

## Layout

- `config.py` — settings: per-user quota, session timeout, rate limit, CORS.
- `models.py` — the `Task` dataclass plus `Priority` and `Status` enums.
- `storage.py` — `InMemoryStore` with quota-aware create and completion.
- `app.py` — Flask routes under `/api/tasks` and a `/healthz` probe.
- `templates/index.html` — the single-page client shell.

## Running

    pip install flask
    python app.py

The API listens on port 5000. Set `TASKLET_DEBUG=1` for the reloader.

## Behavior notes

- Each user is capped at `MAX_TODOS_PER_USER` active tasks; a create past the cap
  archives the oldest completed task instead of failing outright.
- Idle sessions expire after `SESSION_TIMEOUT_MINUTES`; the auth middleware checks
  the issued-at claim on every request.
- Listing returns tasks sorted by priority (CRITICAL first) then creation time.
