"""Gradio UI for the EVOKE live demo.

Connects to two backend processes:
  - port 8000: EVOKE with kv_restore recovery
  - port 8001: truncate policy (evict, no recovery) as the comparison arm

After each assistant turn, polls /health to fetch live cache stats.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator

import gradio as gr
import httpx

EVOKE_URL = os.environ.get("EVOKE_SERVER_URL", "http://127.0.0.1:8000")
DISCARD_URL = os.environ.get("DISCARD_SERVER_URL", "http://127.0.0.1:8001")

_INTRO = """
### How the demo works

Type a question below. The model runs under a **KV budget of 2048 tokens**, so after a few
turns the cache fills and eviction starts firing. With **EVOKE (kv_restore)** selected, evicted
blocks are saved to RAM and spliced back in on demand — the model answers as if they were still
resident. With **Evict, no recovery** selected, evicted tokens are gone and the model must work
from whatever remains in the active cache.

Watch the stat counters at the bottom update after each turn.
"""

_SYSTEM = (
    "You are a helpful assistant. The user will ask questions about a document. "
    "Answer accurately and concisely."
)

_DEMO_DOCUMENT = """
=== EVOKE Research Notes ===

Project: EVOKE (EVict and recOver KV cache Entries)
Goal: Let an LLM inference server handle sessions longer than its physical KV budget by
      evicting and recovering cache blocks on demand.

Key finding: At a KV budget of 1024 tokens, EVOKE recovers 57 % of multi-fact answers
on a 15-seed eval. The best heuristic baseline (H2O) scores 1 % at the same budget.
The gap closes as budget grows: at 4096 tokens all methods approach ceiling.

Architecture: Policy layer in Python. KV save/restore primitives in a forked llama.cpp.
Recovery is identity-keyed (not similarity search). No forward-pass recompute.

Status: Prototype running on NVIDIA RTX 4070 Ti SUPER (Windows, 16 GB VRAM).
"""


def _server(arm: str) -> str:
    return EVOKE_URL if "EVOKE" in arm else DISCARD_URL


def _for_display(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") != "system"]


async def _fetch_stats(base: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{base}/health")
            if r.status_code == 200:
                return r.json()
    except httpx.RequestError:
        pass
    return {}


async def respond(
    message: str,
    history: list[dict],
    arm: str,
) -> AsyncGenerator:
    if not message.strip():
        yield _for_display(history), history, 0, 0, 0, 0.0
        return

    base = _server(arm)

    # Prepend system + demo document on the first user turn.
    if not history:
        messages = [
            {"role": "system", "content": _SYSTEM + "\n\n" + _DEMO_DOCUMENT},
            {"role": "user", "content": message},
        ]
    else:
        messages = list(history) + [{"role": "user", "content": message}]

    partial = ""

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{base}/v1/chat/completions",
                json={
                    "model": "evoke",
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 1024,
                },
            ) as resp:
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                        delta = obj["choices"][0]["delta"].get("content", "")
                        if delta:
                            partial += delta
                            mid = messages + [{"role": "assistant", "content": partial}]
                            yield (
                                _for_display(mid),
                                mid,
                                gr.update(),
                                gr.update(),
                                gr.update(),
                                gr.update(),
                            )
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
    except httpx.RequestError as exc:
        partial = f"[server error: {exc}]"

    final = messages + [{"role": "assistant", "content": partial}]
    stats = await _fetch_stats(base)
    yield (
        _for_display(final),
        final,
        stats.get("total_evictions", 0),
        stats.get("total_recoveries", 0),
        stats.get("total_new_decoded", 0),
        round(stats.get("budget_utilization", 0.0) * 100, 1),
    )


def clear_on_policy_change(_arm: str) -> tuple:
    return [], [], 0, 0, 0, 0.0


async def wait_for_servers(timeout: float = 120.0) -> None:
    for base in (EVOKE_URL, DISCARD_URL):
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    r = await client.get(f"{base}/health")
                    if r.status_code == 200:
                        break
            except httpx.RequestError:
                pass
            await asyncio.sleep(2.0)


with gr.Blocks(title="EVOKE — Live KV Cache Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## EVOKE: Live KV Cache Recovery Demo")
    gr.Markdown(_INTRO)

    arm = gr.Radio(
        choices=["EVOKE (kv_restore)", "Evict, no recovery"],
        value="EVOKE (kv_restore)",
        label="Recovery policy",
        info="Switching clears the current conversation.",
    )

    history_state = gr.State([])

    chatbot = gr.Chatbot(
        type="messages",
        label="Conversation",
        height=480,
        show_copy_button=True,
    )

    msg_box = gr.Textbox(
        placeholder="Ask a question about the demo document...",
        label="Your message",
        lines=2,
        submit_btn=True,
    )

    with gr.Row():
        evictions_box = gr.Number(label="KV evictions", value=0, interactive=False)
        splices_box = gr.Number(
            label="Splices (zero recompute)", value=0, interactive=False
        )
        decoded_box = gr.Number(label="Tokens decoded", value=0, interactive=False)
        budget_box = gr.Number(label="Budget used %", value=0.0, interactive=False)

    submit_event = msg_box.submit(
        fn=respond,
        inputs=[msg_box, history_state, arm],
        outputs=[
            chatbot,
            history_state,
            evictions_box,
            splices_box,
            decoded_box,
            budget_box,
        ],
    )
    submit_event.then(fn=lambda: "", outputs=[msg_box])

    arm.change(
        fn=clear_on_policy_change,
        inputs=[arm],
        outputs=[
            chatbot,
            history_state,
            evictions_box,
            splices_box,
            decoded_box,
            budget_box,
        ],
    )

    gr.Markdown(
        "_Stats update after each turn. "
        "Splice count = KV blocks restored from RAM without forward-pass recompute._"
    )


if __name__ == "__main__":
    demo.launch(
        server_port=int(os.environ.get("GRADIO_PORT", "7860")), server_name="0.0.0.0"
    )
