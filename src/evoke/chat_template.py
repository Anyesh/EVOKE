from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ChatMessage:
    role: str
    content: str


class ChatTemplate:
    def format_messages(
        self, messages: list[ChatMessage], add_generation_prompt: bool = True
    ) -> str:
        raise NotImplementedError

    def wrap_document_and_question(self, document: str, question: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content="Answer questions based on the provided context. Be concise.",
            ),
            ChatMessage(role="user", content=f"{document}\n\nQuestion: {question}"),
        ]
        return self.format_messages(messages, add_generation_prompt=True)

    def wrap_document_prefix(self, document: str) -> str:
        messages = [
            ChatMessage(
                role="system",
                content="Answer questions based on the provided context. Be concise.",
            ),
            ChatMessage(role="user", content=document),
        ]
        return self.format_messages(messages, add_generation_prompt=False)

    def wrap_question_suffix(self, question: str) -> str:
        full = self.wrap_document_and_question("PLACEHOLDER_DOC", question)
        prefix = self.wrap_document_prefix("PLACEHOLDER_DOC")
        return full[len(prefix) :]

    @property
    def stop_strings(self) -> list[str]:
        return []

    @property
    def think_close(self) -> str | None:
        return None

    def extract_answer(self, generated: str) -> str:
        return strip_thinking(generated)


class ChatMLTemplate(ChatTemplate):
    @property
    def stop_strings(self) -> list[str]:
        return ["<|im_end|>"]

    def format_messages(
        self, messages: list[ChatMessage], add_generation_prompt: bool = True
    ) -> str:
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class ChatMLThinkingTemplate(ChatMLTemplate):
    @property
    def think_close(self) -> str | None:
        return "</think>"


class Llama3Template(ChatTemplate):
    @property
    def stop_strings(self) -> list[str]:
        return ["<|eot_id|>"]

    def format_messages(
        self, messages: list[ChatMessage], add_generation_prompt: bool = True
    ) -> str:
        parts = ["<|begin_of_text|>"]
        for msg in messages:
            parts.append(
                f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n"
                f"{msg.content}<|eot_id|>"
            )
        if add_generation_prompt:
            parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)


class PassthroughTemplate(ChatTemplate):
    def format_messages(
        self, messages: list[ChatMessage], add_generation_prompt: bool = True
    ) -> str:
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(msg.content + "\n\n")
            elif msg.role == "user":
                parts.append(msg.content + "\n\n")
            elif msg.role == "assistant":
                parts.append(msg.content)
        return "".join(parts)


_THINK_COMPLETE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
_THINK_INCOMPLETE = re.compile(r"<think>(?:(?!</think>).)*$", re.DOTALL)
_CHAT_STOP = re.compile(r"<\|im_end\|>.*|<\|endoftext\|>.*|<\|eot_id\|>.*", re.DOTALL)


def strip_thinking(text: str) -> str:
    text = _THINK_COMPLETE.sub("", text)
    text = _THINK_INCOMPLETE.sub("", text)
    text = _CHAT_STOP.sub("", text)
    return text.strip()


TEMPLATES: dict[str, type[ChatTemplate]] = {
    "chatml": ChatMLTemplate,
    "llama3": Llama3Template,
    "none": PassthroughTemplate,
}


def detect_template(model_name: str) -> ChatTemplate:
    name = model_name.lower()
    if "qwen" in name:
        if "qwen3" in name or "qwen-3" in name:
            return ChatMLThinkingTemplate()
        return ChatMLTemplate()
    if "llama" in name and "3" in name:
        return Llama3Template()
    if "gemma" in name:
        return ChatMLTemplate()
    return PassthroughTemplate()
