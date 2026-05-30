from __future__ import annotations

from flask import Flask, jsonify, request

from config import DEBUG, MAX_TITLE_LENGTH, RATE_LIMIT_PER_MIN, SECRET_KEY
from models import Priority
from storage import InMemoryStore, QuotaError

app = Flask(__name__)
app.config["SECRET_KEY"] = SECRET_KEY
store = InMemoryStore()


def _owner() -> str:
    return request.headers.get("X-User", "anon")


@app.get("/api/tasks")
def list_tasks():
    return jsonify([t.to_dict() for t in store.list(_owner())])


@app.post("/api/tasks")
def create_task():
    body = request.get_json(force=True) or {}
    title = (body.get("title") or "").strip()
    if not title or len(title) > MAX_TITLE_LENGTH:
        return jsonify({"error": "invalid title"}), 400
    priority = Priority[body.get("priority", "MEDIUM")]
    try:
        task = store.create(_owner(), title, priority)
    except QuotaError as exc:
        return jsonify({"error": str(exc)}), 429
    return jsonify(task.to_dict()), 201


@app.post("/api/tasks/<task_id>/complete")
def complete_task(task_id: str):
    if not store.complete(_owner(), task_id):
        return jsonify({"error": "not found"}), 404
    return jsonify({"status": "done"})


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True, "rate_limit_per_min": RATE_LIMIT_PER_MIN})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=DEBUG)
