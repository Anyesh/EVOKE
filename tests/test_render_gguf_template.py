import os

from evoke.templates import render_gguf_chat_template

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "qwen35_chat_template.jinja"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write",
            "description": "Write a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["filePath", "content"],
            },
        },
    }
]
MESSAGES = [
    {"role": "system", "content": "You are an agent."},
    {"role": "user", "content": "build it"},
]


def _template() -> str:
    with open(FIXTURE, encoding="utf-8") as fh:
        return fh.read()


class TestRenderGgufChatTemplate:
    def test_tools_first_with_function_xml_instructions(self):
        out = render_gguf_chat_template(_template(), MESSAGES, TOOLS)
        assert out.index("# Tools") < out.index("You are an agent.")
        assert "<function=example_function_name>" in out
        assert '"name": "write"' in out

    def test_thinking_prefill_default(self):
        out = render_gguf_chat_template(_template(), MESSAGES, TOOLS)
        assert out.endswith("<|im_start|>assistant\n<think>\n")

    def test_enable_thinking_false_prefills_closed_think(self):
        out = render_gguf_chat_template(
            _template(), MESSAGES, TOOLS, enable_thinking=False
        )
        assert out.endswith("<think>\n\n</think>\n\n")

    def test_assistant_tool_call_echo_renders_function_xml(self):
        msgs = MESSAGES + [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": {"filePath": "a.py", "content": "x = 1"},
                        },
                    }
                ],
            },
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": "continue"},
        ]
        out = render_gguf_chat_template(_template(), msgs, TOOLS)
        assert "<function=write>" in out
        assert "<parameter=filePath>" in out
        assert "<tool_response>\nok\n</tool_response>" in out

    def test_string_tool_call_arguments_normalized(self):
        msgs = MESSAGES + [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "write",
                            "arguments": '{"filePath": "a.py", "content": "x = 1"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "ok"},
            {"role": "user", "content": "continue"},
        ]
        out = render_gguf_chat_template(_template(), msgs, TOOLS)
        assert "<parameter=filePath>" in out
        assert "x = 1" in out

    def test_unrenderable_template_raises_runtime_error(self):
        try:
            render_gguf_chat_template("{{ undefined_name.attr }}", MESSAGES, TOOLS)
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError")
