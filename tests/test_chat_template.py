from evoke.chat_template import (
    ChatMLTemplate,
    Llama3Template,
    PassthroughTemplate,
    detect_template,
    strip_thinking,
)


class TestStripThinking:
    def test_strips_think_block(self):
        text = "<think>\nLet me reason about this.\n</think>\nThe answer is 42."
        assert strip_thinking(text) == "The answer is 42."

    def test_strips_multiple_think_blocks(self):
        text = "<think>first</think> hello <think>second</think> world"
        assert strip_thinking(text) == "hello world"

    def test_no_think_block_unchanged(self):
        text = "The answer is 42."
        assert strip_thinking(text) == "The answer is 42."

    def test_empty_after_strip(self):
        text = "<think>only thinking</think>"
        assert strip_thinking(text) == ""

    def test_strips_incomplete_think_block(self):
        text = "<think>\nStill reasoning about this and ran out of"
        assert strip_thinking(text) == ""

    def test_strips_im_end_and_after(self):
        text = "The answer is 42.<|im_end|>\n<|im_start|>user\nMore stuff"
        assert strip_thinking(text) == "The answer is 42."

    def test_strips_endoftext_and_after(self):
        text = "CRYSTALLINE-HORIZON-42<|endoftext|>junk"
        assert strip_thinking(text) == "CRYSTALLINE-HORIZON-42"

    def test_think_then_answer_then_im_end(self):
        text = "<think>\nreasoning\n</think>\nThe answer is 42.<|im_end|>\njunk"
        assert strip_thinking(text) == "The answer is 42."


class TestChatMLTemplate:
    def test_format_messages(self):
        t = ChatMLTemplate()
        formatted = t.wrap_document_and_question("some doc", "what is it?")
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "some doc" in formatted
        assert "what is it?" in formatted

    def test_document_prefix_no_assistant(self):
        t = ChatMLTemplate()
        prefix = t.wrap_document_prefix("some doc")
        assert "<|im_start|>assistant" not in prefix
        assert "some doc" in prefix

    def test_question_suffix_has_assistant(self):
        t = ChatMLTemplate()
        suffix = t.wrap_question_suffix("what?")
        assert "<|im_start|>assistant" in suffix
        assert "what?" in suffix


class TestLlama3Template:
    def test_format_messages(self):
        t = Llama3Template()
        formatted = t.wrap_document_and_question("doc", "q?")
        assert "<|begin_of_text|>" in formatted
        assert "<|start_header_id|>assistant<|end_header_id|>" in formatted


class TestPassthroughTemplate:
    def test_no_special_tokens(self):
        t = PassthroughTemplate()
        formatted = t.wrap_document_and_question("doc text", "question?")
        assert "<|" not in formatted
        assert "doc text" in formatted
        assert "question?" in formatted


class TestDetectTemplate:
    def test_qwen_gets_chatml(self):
        assert type(detect_template("Qwen2.5-7B-Instruct")).__name__ == "ChatMLTemplate"

    def test_qwen3_gets_chatml(self):
        assert type(detect_template("Qwen3.5-9B-Q4_K_M")).__name__ == "ChatMLTemplate"

    def test_llama3_gets_llama3(self):
        assert type(detect_template("Llama-3.1-8B")).__name__ == "Llama3Template"

    def test_unknown_gets_passthrough(self):
        assert (
            type(detect_template("some-random-model")).__name__ == "PassthroughTemplate"
        )
