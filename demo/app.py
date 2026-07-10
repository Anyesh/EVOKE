from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncGenerator

import gradio as gr
import httpx

EVOKE_URL = os.environ.get("EVOKE_SERVER_URL", "http://127.0.0.1:8000")
DISCARD_URL = os.environ.get("DISCARD_SERVER_URL", "http://127.0.0.1:8001")
JLENS_URL = os.environ.get("JLENS_SERVER_URL", "http://127.0.0.1:8002")

_SYSTEM = (
    "You are a helpful assistant in a multi-turn conversation. "
    "Remember everything the user tells you about themselves (name, location, preferences) "
    "and refer back to those facts when asked. "
    "A reference document is included below for factual questions about EVOKE. "
    "Answer concisely."
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

_SUGGESTED = [
    ("1. Plant a personal fact", "Hi! My name is Alex and I live in Berlin."),
    ("2. Ask about EVOKE", "What is EVOKE's key result at a 1024-token budget?"),
    (
        "3. Ask about the architecture",
        "How does kv_restore work without recomputing anything?",
    ),
    ("4. One more question", "What hardware is EVOKE running on?"),
    ("5. Test the memory", "What is my name and where do I live?"),
]


def _server(arm: str) -> str:
    if "workspace" in arm:
        return JLENS_URL
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


async def _set_budget(tokens: int) -> None:
    async with httpx.AsyncClient(timeout=10.0) as client:
        for base in (EVOKE_URL, DISCARD_URL, JLENS_URL):
            try:
                await client.post(f"{base}/admin/set_budget", json={"tokens": tokens})
            except httpx.RequestError:
                pass


async def respond(
    message: str,
    history: list[dict],
    arm: str,
) -> AsyncGenerator:
    if not message.strip():
        yield _for_display(history), history, 0, 0, 0, 0.0
        return

    base = _server(arm)

    if not history:
        messages = [
            {"role": "system", "content": _SYSTEM + "\n\n" + _DEMO_DOCUMENT},
            {"role": "user", "content": message},
        ]
    else:
        messages = list(history) + [{"role": "user", "content": message}]

    partial = ""

    try:
        # Generous read timeout: on Space CPU hardware a turn can run for
        # minutes, and the server spaces keepalives up to 10s apart during
        # prefill, so a tight read window would abort healthy streams.
        timeout = httpx.Timeout(30.0, read=300.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream(
                "POST",
                f"{base}/v1/chat/completions",
                json={
                    "model": "evoke",
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 512,
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


async def on_budget_change(tokens: int) -> tuple:
    await _set_budget(tokens)
    return [], [], 0, 0, 0, 0.0


async def wait_for_servers(timeout: float = 120.0) -> None:
    for base in (EVOKE_URL, DISCARD_URL, JLENS_URL):
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


_OUTPUTS = None

with gr.Blocks(title="EVOKE: Live KV Cache Recovery") as demo:
    gr.Markdown(
        "# EVOKE: Live KV Cache Recovery\n"
        "Chat with Qwen3-4B running under a tight KV budget. "
        "The **EVOKE** arm saves evicted cache blocks to RAM and splices them back on demand. "
        "The **EVOKE + workspace eviction** arm additionally scores blocks with a probe "
        "distilled from the model's Jacobian-lens workspace, so content the model will "
        "read from later tends not to be evicted in the first place. "
        "The **Evict, no recovery** arm just loses evicted blocks. "
        "Follow the suggested flow below to see the difference."
    )

    with gr.Row():
        arm = gr.Radio(
            choices=[
                "EVOKE (kv_restore)",
                "EVOKE + workspace eviction",
                "Evict, no recovery",
            ],
            value="EVOKE (kv_restore)",
            label="Recovery policy",
            info="Switching resets the conversation.",
        )
        budget_dd = gr.Dropdown(
            choices=[256, 384, 512],
            value=384,
            label="KV budget (tokens)",
            info="Lower = evictions fire sooner. At 512 the chat fits without eviction; "
            "256-384 forces the contrast. Changing resets the conversation.",
        )

    gr.Markdown(
        "**Suggested flow** (click to send, then try step 5 to see if the model remembers):"
    )
    with gr.Row():
        pill_buttons = [
            gr.Button(label, size="sm", variant="secondary") for label, _ in _SUGGESTED
        ]

    history_state = gr.State([])

    chatbot = gr.Chatbot(label="Conversation", height=420)

    msg_box = gr.Textbox(
        placeholder="Type a message or click a step above...",
        label="Your message",
        lines=2,
        submit_btn=True,
    )

    with gr.Row():
        evictions_box = gr.Number(label="KV evictions", value=0, interactive=False)
        recoveries_box = gr.Number(
            label="KV recoveries (zero recompute)", value=0, interactive=False
        )
        decoded_box = gr.Number(label="Tokens decoded", value=0, interactive=False)
        budget_box = gr.Number(label="Budget used %", value=0.0, interactive=False)

    _all_outputs = [
        chatbot,
        history_state,
        evictions_box,
        recoveries_box,
        decoded_box,
        budget_box,
    ]

    submit_event = msg_box.submit(
        fn=respond,
        inputs=[msg_box, history_state, arm],
        outputs=_all_outputs,
    )
    submit_event.then(fn=lambda: "", outputs=[msg_box])

    def make_pill_handler(prompt: str):
        async def handler(history, arm_val):
            async for chunk in respond(prompt, history, arm_val):
                yield chunk

        return handler

    for btn, (_, prompt) in zip(pill_buttons, _SUGGESTED):
        btn.click(
            fn=make_pill_handler(prompt),
            inputs=[history_state, arm],
            outputs=_all_outputs,
        )

    arm.change(fn=clear_on_policy_change, inputs=[arm], outputs=_all_outputs)
    budget_dd.change(fn=on_budget_change, inputs=[budget_dd], outputs=_all_outputs)

    with gr.Accordion("What's in the reference document?", open=False):
        gr.Markdown(
            "The model has access to this document as a system-level context block. "
            "It is the first thing that gets evicted as the conversation grows.\n\n"
            "```\n" + _DEMO_DOCUMENT.strip() + "\n```"
        )

    gr.Markdown(
        "_Stats update after each turn. "
        "Recoveries = KV blocks spliced back from RAM, zero forward-pass recompute. "
        "Switch to **Evict, no recovery** after a few turns to see what the model forgets._"
    )

    gr.Markdown(
        "**Links:** "
        "[GitHub](https://github.com/Anyesh/EVOKE) "
        "| [Paper (PDF)](https://github.com/Anyesh/EVOKE/blob/main/paper/paper.pdf) "
        "| [llama.cpp fork](https://github.com/Anyesh/llama.cpp)"
    )


if __name__ == "__main__":
    demo.launch(
        server_port=int(os.environ.get("GRADIO_PORT", "7860")),
        server_name="0.0.0.0",
    )
