"""Local LLM judge for benchmark tiebreak scoring.

Wraps a second llama.cpp engine (typically a smaller, retrieval-grade model
loaded sequentially after the system-under-test) and exposes a yes/no judging
call. Used by the multi-fact benchmark to break ambiguous answers where
string-match-set failed but the model's response contains semantically related
content (paraphrased the value rather than emitting it verbatim, partial
match, formatting variance the matcher missed). The judge is not invoked on
unambiguous matches or unambiguous misses, so its cost stays bounded.

Design constraints:
- Different model family from the system under test (avoid Qwen-judges-Qwen
  circularity). Default is Gemma 4 E4B Q4_K_M (~4.6 GiB).
- Sequential loading on a single 16 GiB GPU is the locked deployment shape;
  callers must unload the SUT (engine.close()) before constructing the judge,
  and unload the judge before resuming the SUT.
- Greedy decoding with a very short answer budget; the prompt asks for a
  single YES or NO token so latency is in the tens of milliseconds per call
  regardless of fact length.
"""

from __future__ import annotations

import os
from pathlib import Path

from evoke.llama_engine import LlamaCppEngine


_DEFAULT_JUDGE_PATH = (
    r"C:\paths\llama-cpp\models\gguf\gemma-4-E4B-it-Q4_K_M.gguf"
)

_JUDGE_PROMPT = """You are a strict but fair judge. Given a planted fact and a model's answer, decide whether the answer correctly recalls the fact.

Reply with exactly one word: YES if the answer correctly recovers the fact's content (paraphrases are acceptable), or NO if the answer is missing the fact or recalls a wrong value.

Fact: {fact}
Answer: {answer}

Decision (YES or NO):"""


class LLMJudge:
    def __init__(
        self,
        model_path: str | None = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        verbose: bool = False,
    ) -> None:
        self._model_path = model_path or os.environ.get(
            "EVOKE_MFB_JUDGE_PATH", _DEFAULT_JUDGE_PATH
        )
        if not Path(self._model_path).exists():
            raise FileNotFoundError(
                f"judge model not found at {self._model_path}; set "
                "EVOKE_MFB_JUDGE_PATH or pass model_path explicitly"
            )
        self._engine = LlamaCppEngine(
            self._model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=verbose,
        )

    def score(self, fact: str, answer: str) -> bool:
        prompt = _JUDGE_PROMPT.format(fact=fact, answer=answer)
        self._engine.reset()
        tokens = self._engine.tokenize(prompt)
        self._engine.process_tokens(tokens)
        # 8 tokens is enough for "YES" or "NO" plus surrounding punctuation
        # variants the model might emit; the parser below trims whitespace
        # and looks for the first definitive token.
        output_tokens: list[int] = []
        eos = self._engine.eos_token
        for _ in range(8):
            tok = self._engine.generate_next()
            output_tokens.append(tok)
            if tok == eos:
                break
        text = self._engine.detokenize(output_tokens).strip().upper()
        # First token settles the verdict. "YES", "Y", "TRUE", "1" -> True;
        # anything else -> False. Default-to-False so an ambiguous judge
        # output never inflates the pass-rate.
        first_word = text.split()[0] if text else ""
        first_word = first_word.strip(".,!?:;\"'`")
        return first_word in {"YES", "Y", "TRUE", "1", "CORRECT"}

    def close(self) -> None:
        self._engine.close()

    def __enter__(self) -> LLMJudge:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
