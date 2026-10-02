from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class RequestDumper:
    """Writes selected chat requests and their rendered prompts to files, for checking what the model was sent."""

    def __init__(self, directory: str | Path, first: int = 3, every: int = 25) -> None:
        self.directory = Path(directory)
        self.first = first
        self.every = every
        self._count = 0

    def maybe_dump(self, payload: dict[str, Any]) -> bool:
        self._count += 1
        if not (
            self._count <= self.first
            or (self.every > 0 and self._count % self.every == 0)
        ):
            return False
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"request-{self._count:06d}.json"
        path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
        return True


def from_env() -> RequestDumper | None:
    directory = os.environ.get("EVOKE_DUMP_REQUESTS_DIR")
    if not directory:
        return None
    return RequestDumper(
        directory,
        first=int(os.environ.get("EVOKE_DUMP_FIRST", "3")),
        every=int(os.environ.get("EVOKE_DUMP_EVERY", "25")),
    )
