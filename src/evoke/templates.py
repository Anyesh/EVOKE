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

from jinja2 import Environment, StrictUndefined
from jinja2.exceptions import TemplateError

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)
_FUNCTION_RE = re.compile(
    r"<function=(?P<name>[\w.\-]+)>\s*(?P<body>.*?)\s*</function>", re.DOTALL
)
_PARAMETER_RE = re.compile(
    r"<parameter=(?P<key>[\w.\-]+)>\n?(?P<value>.*?)\n?</parameter>", re.DOTALL
)
# Generation can end (stop string, EOS, token budget) before the model closes
# its tags, so unterminated variants must parse too or the client receives an
# empty message for an otherwise valid call.
_FUNCTION_OPEN_RE = re.compile(
    r"<function=(?P<name>[\w.\-]+)>\s*(?P<body>.*)\Z", re.DOTALL
)
_PARAMETER_OPEN_RE = re.compile(
    r"<parameter=(?P<key>[\w.\-]+)>\n?(?P<value>.*)\Z", re.DOTALL
)
_CLOSE_TAGS = ("</tool_call>", "</function>", "</parameter>")


def _strip_partial_close(text: str) -> str:
    # A truncated generation can dangle a fragment of a close tag ("</tool_",
    # "</param"); strip strict prefixes only, because a full close tag means
    # the block was not truncated there.
    out = text.rstrip()
    for tag in _CLOSE_TAGS:
        for n in range(len(tag) - 1, 1, -1):
            if out.endswith(tag[:n]):
                return out[:-n].rstrip()
    return out


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


def _jinja_raise(message: str) -> None:
    # raise_exception is a Hermes/Qwen template convention used inside the
    # Jinja template to abort on malformed inputs. Minja (llama.cpp's C++
    # Jinja parser) exposes it; we mirror the semantics here so Python-side
    # rendering of the same templates respects the same contract.
    raise RuntimeError(f"chat template raise_exception: {message}")


def _normalize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # OpenAI clients echo assistant tool_calls with arguments as a JSON
    # string, but HF-convention templates iterate arguments as a mapping
    # (e.g. Qwen3.5's `arguments|items`), so string arguments must be
    # decoded before rendering. A tool-call echo also carries content=null,
    # which pydantic's exclude_none dump drops entirely; GGUF templates
    # access message.content unconditionally and this renderer uses
    # StrictUndefined, so the key must exist or the render falls back to
    # format_qwen_chat, whose differently-formatted tools JSON breaks the
    # byte-stable prefix across turns.
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.get("content") is None:
            m = {**m, "content": ""}
        tcs = m.get("tool_calls")
        if not tcs:
            out.append(m)
            continue
        fixed_tcs = []
        for tc in tcs:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args, strict=False)
                except json.JSONDecodeError:
                    args = {"_raw": args}
                tc = {**tc, "function": {**fn, "arguments": args}}
            fixed_tcs.append(tc)
        out.append({**m, "tool_calls": fixed_tcs})
    return out


def render_gguf_chat_template(
    template_str: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    add_generation_prompt: bool = True,
    enable_thinking: bool | None = None,
) -> str:
    """Render a GGUF-embedded Jinja chat template with tool support.

    enable_thinking=None leaves the variable undefined so the template's own
    default applies; templates differ on which branch is the default (the
    Qwen3.5 GGUF variant defaults thinking off, the HF repo variant defaults
    it on), so callers must opt in explicitly per model.
    """
    env = Environment(
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
        extensions=["jinja2.ext.do", "jinja2.ext.loopcontrols"],
    )
    env.globals["raise_exception"] = _jinja_raise
    render_kwargs: dict[str, Any] = {
        "messages": _normalize_messages(messages),
        "tools": tools,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        render_kwargs["enable_thinking"] = enable_thinking
    try:
        template = env.from_string(template_str)
        return template.render(**render_kwargs)
    except TemplateError as exc:
        raise RuntimeError(f"jinja template render failed: {exc}") from exc


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


def _tool_param_types(tools: list[dict[str, Any]] | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for t in tools or []:
        fn = t.get("function") or {}
        name = fn.get("name")
        if not name:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        out[name] = {
            k: (v.get("type", "") if isinstance(v, dict) else "")
            for k, v in props.items()
        }
    return out


def _coerce_param(value: str, ptype: str) -> Any:
    # Without a schema (or for string params) the raw text is the value;
    # coercing JSON-looking strings would corrupt string params that happen
    # to contain JSON (e.g. writing a .json file).
    if ptype in ("", "string"):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_call_body(
    body: str, param_types: dict[str, dict[str, str]]
) -> ParsedToolCall | None:
    if body.startswith("{"):
        try:
            # strict=False because models emit literal newlines and tabs
            # inside long string values (file contents), which strict JSON
            # rejects.
            obj = json.loads(body, strict=False)
        except json.JSONDecodeError:
            return None
        name = obj.get("name", "")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args, strict=False)
            except json.JSONDecodeError:
                args = {"_raw": args}
        return ParsedToolCall(
            id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args
        )

    fn = _FUNCTION_RE.search(body)
    if fn is not None:
        name = fn.group("name")
        fbody = fn.group("body")
    else:
        fn_open = _FUNCTION_OPEN_RE.search(body)
        if fn_open is None:
            return None
        name = fn_open.group("name")
        fbody = fn_open.group("body")
    types = param_types.get(name, {})
    args = {}
    last_end = 0
    for pm in _PARAMETER_RE.finditer(fbody):
        key = pm.group("key")
        args[key] = _coerce_param(pm.group("value"), types.get(key, ""))
        last_end = pm.end()
    pm_open = _PARAMETER_OPEN_RE.search(fbody[last_end:])
    if pm_open is not None and pm_open.group("key") not in args:
        key = pm_open.group("key")
        args[key] = _coerce_param(
            _strip_partial_close(pm_open.group("value")), types.get(key, "")
        )
    return ParsedToolCall(id=f"call_{uuid.uuid4().hex[:12]}", name=name, arguments=args)


def parse_qwen_response(
    raw: str,
    *,
    strip_thinking: bool = True,
    tools: list[dict[str, Any]] | None = None,
) -> ParsedResponse:
    """Split a Qwen assistant turn into plain content and structured tool calls.

    By default strips a <think>...</think> reasoning trace (opencode does not
    need to see internal reasoning). Pass strip_thinking=False when running
    against hybrid memory models that can't evict mid-cache: the cached state
    must include the thinking trace, and so must the response, otherwise the
    next request's templated prompt diverges and triggers a session reset.
    """
    cleaned = raw
    if strip_thinking:
        think_end = cleaned.find("</think>")
        if think_end != -1:
            cleaned = cleaned[think_end + len("</think>") :].lstrip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        idx = cleaned.find(token)
        if idx != -1:
            cleaned = cleaned[:idx]

    param_types = _tool_param_types(tools)
    tool_calls: list[ParsedToolCall] = []
    content_parts: list[str] = []
    cursor = 0
    for m in _TOOL_CALL_RE.finditer(cleaned):
        if m.start() > cursor:
            content_parts.append(cleaned[cursor : m.start()])
        call = _parse_tool_call_body(m.group("body"), param_types)
        if call is not None:
            tool_calls.append(call)
        else:
            content_parts.append(m.group(0))
        cursor = m.end()
    if cursor < len(cleaned):
        remainder = cleaned[cursor:]
        open_idx = remainder.find("<tool_call>")
        if open_idx != -1 and "</tool_call>" not in remainder[open_idx:]:
            # The close tag never arrived (stop string or budget cut the
            # generation), so parse the unterminated block rather than
            # returning an empty message for a recoverable call.
            body = _strip_partial_close(
                remainder[open_idx + len("<tool_call>") :].strip()
            )
            call = _parse_tool_call_body(body, param_types)
            if call is not None:
                tool_calls.append(call)
                remainder = remainder[:open_idx]
        content_parts.append(remainder)

    content = "".join(content_parts).strip()
    finish = "tool_calls" if tool_calls else "stop"
    return ParsedResponse(content=content, tool_calls=tool_calls, finish_reason=finish)
