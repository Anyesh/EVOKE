"""Chat-template formatting and tool-call parsing for Qwen-family models.

The OpenAI-compatible chat completions API exchanges structured messages and
tool definitions; the underlying model only sees raw text. This module is the
boundary: it renders the OpenAI message list into the model's expected raw
prompt, and parses the model's raw output back into structured tool calls.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from typing import Any

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>\{.*?\})\s*</tool_call>", re.DOTALL)


@dataclass
class ParsedToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedResponse:
    content: str
    tool_calls: list[ParsedToolCall]
    finish_reason: str


def _render_tools(tools: list[dict[str, Any]] | None) -> str:
    if not tools:
        return ""
    spec = json.dumps(tools, indent=2)
    return (
        "\n\n# Tools\n\n"
        "You may call one or more functions to assist with the user query.\n\n"
        "You are provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>\n" + spec + "\n</tools>\n\n"
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:\n"
        "<tool_call>\n"
        '{"name": <function-name>, "arguments": <args-json-object>}\n'
        "</tool_call>"
    )


def format_qwen_chat(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """Render OpenAI-style messages into Qwen's <|im_start|>/<|im_end|> format.

    System message is augmented with tool definitions when tools are present. Tool
    results from the agent come back as messages with role="tool" and are mapped
    into Qwen's <tool_response> envelope.
    """
    parts: list[str] = []
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    system_content = "\n".join(m.get("content", "") for m in system_msgs)
    if tools:
        system_content = (
            system_content or "You are a helpful assistant."
        ) + _render_tools(tools)
    if system_content:
        parts.append(f"<|im_start|>system\n{system_content}<|im_end|>")

    for m in other_msgs:
        role = m.get("role", "user")
        content = m.get("content") or ""
        if role == "tool":
            tool_name = m.get("name", "")
            wrapped = f"<tool_response>\n{content}\n</tool_response>"
            parts.append(f"<|im_start|>user\n{wrapped}<|im_end|>")
            continue
        if role == "assistant":
            tcs = m.get("tool_calls") or []
            if tcs:
                inner: list[str] = []
                if content:
                    inner.append(content)
                for tc in tcs:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        args_obj = args
                    else:
                        args_obj = json.dumps(args)
                    call = f'<tool_call>\n{{"name": "{name}", "arguments": {args_obj}}}\n</tool_call>'
                    inner.append(call)
                content = "\n".join(inner)
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            continue
        parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")

    rendered = "\n".join(parts)
    if add_generation_prompt:
        rendered += "\n<|im_start|>assistant\n"
    return rendered


def parse_qwen_response(raw: str) -> ParsedResponse:
    """Split a Qwen assistant turn into plain content and structured tool calls.

    Strips a <think>...</think> reasoning trace if present (kept out of the
    returned content; opencode does not need to see internal reasoning).
    """
    cleaned = raw
    think_end = cleaned.find("</think>")
    if think_end != -1:
        cleaned = cleaned[think_end + len("</think>") :].lstrip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        idx = cleaned.find(token)
        if idx != -1:
            cleaned = cleaned[:idx]

    tool_calls: list[ParsedToolCall] = []
    content_parts: list[str] = []
    cursor = 0
    for m in _TOOL_CALL_RE.finditer(cleaned):
        if m.start() > cursor:
            content_parts.append(cleaned[cursor : m.start()])
        body = m.group("body")
        try:
            obj = json.loads(body)
            name = obj.get("name", "")
            args = obj.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(
                ParsedToolCall(
                    id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args
                )
            )
        except json.JSONDecodeError:
            content_parts.append(m.group(0))
        cursor = m.end()
    if cursor < len(cleaned):
        content_parts.append(cleaned[cursor:])

    content = "".join(content_parts).strip()
    finish = "tool_calls" if tool_calls else "stop"
    return ParsedResponse(content=content, tool_calls=tool_calls, finish_reason=finish)
