"""Shared trajectory data types for trajectory_bench.py and
trajectory_normalizers.py. Split out because those two modules used to
import from each other directly, which broke the moment trajectory_bench.py
was run as __main__ instead of imported as a package.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ToolCall:
    name: str
    args: dict

    def target(self) -> str:
        for key in ("path", "file", "file_path", "target", "filename"):
            if key in self.args:
                return str(self.args[key])
        return next((str(v) for v in self.args.values()), "")

    def signature_tiers(self) -> tuple[str, str, str]:
        # Coarsest first: exact-args is too strict for free-form model
        # output, tool-name-only is too loose to discriminate policies.
        # Scoring headlines on the middle tier.
        target = self.target()
        args_sig = ",".join(f"{k}={v}" for k, v in sorted(self.args.items()))
        return (self.name, f"{self.name}:{target}", f"{self.name}:{args_sig}")


@dataclass
class Step:
    role: str
    text: str
    tool_call: ToolCall | None = None


@dataclass
class Trajectory:
    traj_id: str
    system_prompt: str
    steps: list[Step]
