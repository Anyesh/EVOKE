"""Corrected trajectory normalizers for nebius/SWE-agent-trajectories and
nebius/SWE-rebench-openhands-trajectories, verified against real rows fetched
from the HuggingFace datasets-server API (not guessed from field names).

trajectory_bench.py's own _normalize_openhands/_normalize_swe_agent were
marked in their comments as unverified. Against real data: the OpenHands
normalizer never actually fires (its dispatch key "history" doesn't exist in
real rows), and OpenHands rows instead fall through to _normalize_swe_agent,
which happens to parse them correctly by accident because OpenHands' export
is already an OpenAI-style {role, content, tool_calls} message list. The
SWE-agent normalizer is a real break: SWE-agent items use {role, text,
system_prompt} with role "ai" for the assistant and no tool_calls field at
all, so every SWE-agent row was silently producing zero tool calls.

To use from trajectory_bench.py, replace its inline key-sniffing dispatch
with a call to this module's normalize(raw, traj_id) function.
"""

from __future__ import annotations

import json
import re
import shlex

from trajectory_types import Step, ToolCall, Trajectory

# The closed command set SWE-agent's own system prompt defines (verified
# against a real system_prompt string: open, goto, scroll_down, scroll_up,
# create, submit, search_dir, search_file, find_file, edit all appear there).
# Anything else in a fenced command block is a raw shell command.
_KNOWN_COMMANDS = {
    "open",
    "goto",
    "scroll_down",
    "scroll_up",
    "create",
    "submit",
    "search_dir",
    "search_file",
    "find_file",
    "edit",
    "exit_cost",
}

_FENCED_BLOCK_RE = re.compile(r"```(?:bash)?\n?(.*?)```", re.DOTALL)


def classify(raw: dict) -> str:
    """Dataset discriminator. Both datasets carry a "trajectory" key, so
    key-sniffing on that alone (trajectory_bench.py's original dispatch)
    can't tell them apart; trajectory_id vs model_name can."""
    if "trajectory_id" in raw:
        return "openhands"
    if "model_name" in raw:
        return "swe_agent"
    raise RuntimeError(
        f"unrecognized trajectory schema, top-level keys={list(raw.keys())}"
    )


def _tool_call_from_openhands_message(msg: dict) -> ToolCall | None:
    tool_calls = msg.get("tool_calls") or []
    if not tool_calls:
        return None
    fn = tool_calls[0].get("function", {})
    name = fn.get("name")
    if not name:
        return None
    raw_args = fn.get("arguments", "{}")
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {}
    else:
        args = raw_args
    return ToolCall(name=name, args=args if isinstance(args, dict) else {})


def normalize_openhands(raw: dict, traj_id: str) -> Trajectory:
    """OpenHands rows are already an OpenAI-style message list under
    "trajectory", with the system prompt as trajectory[0] (role=system).
    There is no top-level "system"/"instruction" key to read, which is what
    made the original normalizer's system_prompt always come out empty."""
    messages = raw.get("trajectory", [])
    system_prompt = ""
    start = 0
    if messages and messages[0].get("role") == "system":
        system_prompt = str(messages[0].get("content", ""))
        start = 1

    steps: list[Step] = []
    for msg in messages[start:]:
        role = msg.get("role", "assistant")
        content = str(msg.get("content", "") or "")
        call = _tool_call_from_openhands_message(msg) if role == "assistant" else None
        steps.append(Step(role=role, text=content, tool_call=call))
    return Trajectory(traj_id=traj_id, system_prompt=system_prompt, steps=steps)


def _parse_swe_agent_command(text: str) -> ToolCall | None:
    """SWE-agent has no structured tool_calls field; the model emits its
    command as the last fenced code block in free-form "ai" text. Extract it
    and classify against the DSL commands the system prompt itself defines,
    falling back to a raw-bash ToolCall for anything else."""
    blocks = _FENCED_BLOCK_RE.findall(text)
    if not blocks:
        return None
    block = blocks[-1].strip()
    if not block:
        return None

    first_line = block.splitlines()[0].strip()
    try:
        tokens = shlex.split(first_line)
    except ValueError:
        tokens = first_line.split()
    if not tokens:
        return None

    name = tokens[0]
    if name not in _KNOWN_COMMANDS:
        # Plain shell command (e.g. "ls -F", "pytest tests/"): the whole
        # line is both the command and its own target for revisit purposes.
        return ToolCall(name="bash", args={"cmd": block})

    args: dict = {}
    if name == "open":
        args["path"] = tokens[1] if len(tokens) > 1 else ""
        if len(tokens) > 2:
            args["line_number"] = tokens[2]
    elif name == "goto":
        args["line_number"] = tokens[1] if len(tokens) > 1 else ""
    elif name == "create":
        args["filename"] = tokens[1] if len(tokens) > 1 else ""
    elif name == "find_file":
        args["filename"] = tokens[1] if len(tokens) > 1 else ""
        if len(tokens) > 2:
            args["dir"] = tokens[2]
    elif name == "search_dir":
        args["search_term"] = tokens[1] if len(tokens) > 1 else ""
        if len(tokens) > 2:
            args["dir"] = tokens[2]
    elif name == "search_file":
        args["search_term"] = tokens[1] if len(tokens) > 1 else ""
        if len(tokens) > 2:
            args["file"] = tokens[2]
    elif name == "edit":
        # No path in the command itself; edit implicitly targets whatever
        # file is currently open in the editor, which this normalizer has
        # no state to recover. Known limitation: edit steps won't register
        # a meaningful target for revisit detection.
        args["range"] = tokens[1] if len(tokens) > 1 else ""
    # scroll_down / scroll_up / submit / exit_cost take no arguments.

    return ToolCall(name=name, args=args)


def normalize_swe_agent(raw: dict, traj_id: str) -> Trajectory:
    """SWE-agent items are {cutoff_date, mask, role, system_prompt, text}:
    role "ai" (not "assistant") is the model, the actual text lives in
    "text" (system items instead carry it in "system_prompt", with
    text=None), and there's no tool_calls field, so commands must be parsed
    out of fenced code blocks in the "ai" text. The first "user" item is the
    real task instruction; every subsequent "user" item is environment
    output (bash stdout, tracebacks) and is remapped to role "tool" to match
    the schema the harness expects."""
    items = raw.get("trajectory", [])
    system_prompt = ""
    start = 0
    if items and items[0].get("role") == "system":
        system_prompt = str(items[0].get("system_prompt") or "")
        start = 1

    steps: list[Step] = []
    seen_user = False
    for item in items[start:]:
        role = item.get("role", "ai")
        text = str(item.get("text") or "")
        if role == "ai":
            steps.append(
                Step(
                    role="assistant",
                    text=text,
                    tool_call=_parse_swe_agent_command(text),
                )
            )
        elif role == "user":
            if not seen_user:
                steps.append(Step(role="user", text=text))
                seen_user = True
            else:
                steps.append(Step(role="tool", text=text))
        else:
            steps.append(Step(role=role, text=text))
    return Trajectory(traj_id=traj_id, system_prompt=system_prompt, steps=steps)


def normalize(raw: dict, traj_id: str) -> Trajectory:
    kind = classify(raw)
    if kind == "openhands":
        return normalize_openhands(raw, traj_id)
    return normalize_swe_agent(raw, traj_id)
