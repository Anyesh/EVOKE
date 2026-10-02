"""The rendered prompt must match what transformers produces for the same template.

The golden files were produced with transformers' render_jinja_template (the code behind
apply_chat_template) from the fixture templates and the conversation below. Regenerate with
the same call, not with EVOKE's renderer. When transformers is installed the test also renders
live and compares.
"""

import json
from pathlib import Path

import pytest

from evoke.templates import render_gguf_chat_template

FIXTURES = Path(__file__).resolve().parent / "fixtures"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file; don't use for <dirs> & more, café",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "it's the path"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
    },
]
MESSAGES = [
    {"role": "system", "content": "You're an agent. Use <tools> & be careful."},
    {"role": "user", "content": "Fix the bug in 'a.py' where x < y && z > w."},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read", "arguments": '{"path": "a.py", "limit": 50}'},
            }
        ],
    },
    {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "read",
        "content": "if x < y and z > w: print('it\\'s ok') # a & b",
    },
    {
        "role": "assistant",
        "content": "Looking.",
        "tool_calls": [
            {
                "id": "c2",
                "type": "function",
                "function": {"name": "glob", "arguments": {"pattern": "*.py"}},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "c2", "name": "glob", "content": "a.py\nb.py"},
    {"role": "user", "content": "thanks"},
]
NAMES = ["qwen3", "qwen35", "qwen36"]


def _evoke(name):
    template = (FIXTURES / f"{name}_chat_template.jinja").read_text()
    return render_gguf_chat_template(
        template, MESSAGES, TOOLS, add_generation_prompt=True, enable_thinking=False
    )


@pytest.mark.parametrize("name", NAMES)
def test_render_equals_the_transformers_golden(name):
    golden = (FIXTURES / f"{name}_render_golden.txt").read_text()
    assert _evoke(name) == golden


@pytest.mark.parametrize("name", NAMES)
def test_render_equals_live_transformers_when_installed(name):
    utils = pytest.importorskip("transformers.utils.chat_template_utils")
    template = (FIXTURES / f"{name}_chat_template.jinja").read_text()
    hf_messages = json.loads(json.dumps(MESSAGES))
    for message in hf_messages:
        for call in message.get("tool_calls") or []:
            if isinstance(call["function"]["arguments"], str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    rendered, _ = utils.render_jinja_template(
        conversations=[hf_messages],
        tools=TOOLS,
        chat_template=template,
        add_generation_prompt=True,
        enable_thinking=False,
        bos_token="",
        eos_token="",
    )
    assert rendered[0] == _evoke(name)
