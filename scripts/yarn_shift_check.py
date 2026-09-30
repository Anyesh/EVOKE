"""A K-shift must be a pure rotation: after evicting mid-cache ranges (so survivors
are shifted), greedy output must match a fresh decode of the same visible tokens.
Fails on a llama.cpp build whose K-shift rescales K by the YaRN magnitude scale.

  python scripts/yarn_shift_check.py MODEL TEXT_FILE FACTOR [N_SHIFTS]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from evoke.llama_engine import LlamaCppEngine

model, text_path, factor = sys.argv[1], sys.argv[2], float(sys.argv[3])
n_shifts = int(sys.argv[4]) if len(sys.argv) > 4 else 6
kwargs = dict(n_ctx=8192, n_gpu_layers=-1, verbose=False)
if factor > 1:
    kwargs.update(yarn_factor=factor, yarn_orig_ctx=40960)


def fresh(tokens, n_gen):
    eng = LlamaCppEngine(model, **kwargs)
    eng.process_tokens(tokens)
    out = [eng.generate_next() for _ in range(n_gen)]
    text = eng.detokenize(out)
    eng.close()
    return text


probe = LlamaCppEngine(model, **kwargs)
all_tokens = probe.tokenize(Path(text_path).read_text())
tokens, tail = all_tokens[:2500], all_tokens[2500:2508]
probe.process_tokens(tokens)
kept = list(tokens)
for k in range(n_shifts):
    pos = 200 + 30 * k
    probe.evict_ranges([(pos, pos + 4)])
    del kept[pos : pos + 4]
probe.process_tokens(tail)
shifted = probe.detokenize([probe.generate_next() for _ in range(25)])
probe.close()
print("shifted   :", repr(shifted))
print("contiguous:", repr(fresh(kept + tail, 25)))
