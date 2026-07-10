import os

from evoke.llama_engine import LlamaCppEngine
from evoke.templates import render_gguf_chat_template

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "qwen35_chat_template.jinja"
)
QWEN3_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "qwen3_chat_template.jinja"
)
LLAMA31_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "llama31_chat_template.jinja"
)
QWEN36_FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "qwen36_chat_template.jinja"
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


def _qwen3_template() -> str:
    with open(QWEN3_FIXTURE, encoding="utf-8") as fh:
        return fh.read()


def _llama31_template() -> str:
    with open(LLAMA31_FIXTURE, encoding="utf-8") as fh:
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

    def test_assistant_echo_without_content_key_renders_qwen3(self):
        # An OpenAI client echoing our tool-call response sends the assistant
        # message with content=null; pydantic model_dump(exclude_none=True)
        # then drops the key entirely. The Qwen3 GGUF template accesses
        # message.content unconditionally, so a missing key must not push the
        # render onto the format_qwen_chat fallback: the fallback pretty-prints
        # the tools JSON while the GGUF template emits it compact, and that
        # byte drift mid-prompt resets the session and discards every
        # recoverable block.
        msgs = MESSAGES + [
            {
                "role": "assistant",
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
        ]
        out = render_gguf_chat_template(_qwen3_template(), msgs, TOOLS)
        assert "<tool_call>" in out
        assert "ok" in out

    def test_assistant_echo_without_content_key_renders_qwen35(self):
        msgs = MESSAGES + [
            {
                "role": "assistant",
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
        ]
        out = render_gguf_chat_template(_template(), msgs, TOOLS)
        assert "<function=write>" in out

    def test_text_only_assistant_echo_renders_qwen36(self):
        # Live failure (Claude Code via CCR on Qwen3.6-35B): a plain-text
        # assistant echo carries no tool_calls key, and the Qwen3.5/3.6
        # template truth-tests message.tool_calls, which StrictUndefined
        # rejects even inside an `and` chain. The silent format_qwen_chat
        # fallback then byte-diverges from the previous turn's GGUF render
        # and forces a full re-prefill of the 40K-token prompt every turn.
        with open(QWEN36_FIXTURE, encoding="utf-8") as fh:
            tmpl = fh.read()
        msgs = MESSAGES + [
            {"role": "assistant", "content": "I will read the file now."},
            {"role": "user", "content": "go on"},
        ]
        out = render_gguf_chat_template(tmpl, msgs, TOOLS)
        assert "I will read the file now." in out

    def test_bos_eos_tokens_reach_the_template(self):
        out = render_gguf_chat_template(
            "{{ bos_token }}x{{ eos_token }}",
            MESSAGES,
            None,
            bos_token="<BOS>",
            eos_token="<EOS>",
        )
        assert out == "<BOS>x<EOS>"

    def test_bos_token_defaults_empty_instead_of_undefined(self):
        out = render_gguf_chat_template("{{ bos_token }}hi", MESSAGES, None)
        assert out == "hi"

    def test_llama31_template_renders_with_tools(self):
        # Live failure (opencode on Meta-Llama-3.1-8B): the Llama 3.1 GGUF
        # template opens with {{- bos_token }}, which StrictUndefined turned
        # into a render failure, so the server silently fell back to the Qwen
        # ChatML format. Llama has no ChatML special tokens, mimicked the
        # markers as plain text, and streamed a hallucinated multi-turn
        # conversation.
        out = render_gguf_chat_template(
            _llama31_template(),
            MESSAGES,
            TOOLS,
            bos_token="<|begin_of_text|>",
            eos_token="<|eot_id|>",
        )
        assert out.startswith("<|begin_of_text|>")
        assert "<|start_header_id|>" in out
        assert '"name": "write"' in out
        assert "<|im_start|>" not in out

    def test_unrenderable_template_raises_runtime_error(self):
        try:
            render_gguf_chat_template("{{ undefined_name.attr }}", MESSAGES, TOOLS)
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError")


class _StubEngine:
    bos_token = -1
    eos_token = -1

    def __init__(self, template: str):
        self._template = template
        self.c_path_calls = 0

    def get_chat_template_string(self) -> str:
        return self._template

    def detokenize(self, tokens: list[int]) -> str:
        return ""

    def apply_chat_template(
        self, messages: list[dict], add_generation_prompt: bool = True
    ) -> str:
        self.c_path_calls += 1
        return "C-PATH"


class TestEngineTemplateRouting:
    def test_enable_thinking_routes_jinja_even_without_tools(self):
        # The C llama_chat_apply_template API cannot carry enable_thinking,
        # so an explicit flag must route through the jinja renderer or the
        # EVOKE_ENABLE_THINKING env is silently a no-op for tool-less clients.
        stub = _StubEngine(_qwen3_template())
        out = LlamaCppEngine.apply_chat_template_with_tools(
            stub, MESSAGES, None, enable_thinking=False
        )
        assert stub.c_path_calls == 0
        assert out.endswith("<think>\n\n</think>\n\n")

    def test_no_tools_no_flag_keeps_c_path(self):
        stub = _StubEngine(_qwen3_template())
        out = LlamaCppEngine.apply_chat_template_with_tools(stub, MESSAGES, None)
        assert out == "C-PATH"
        assert stub.c_path_calls == 1
