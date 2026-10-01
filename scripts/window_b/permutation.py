import argparse
import hashlib
import json
from pathlib import Path

from schedule import permutation

SEED = 20260927

parser = argparse.ArgumentParser()
parser.add_argument("--ids", default="results/window_b/instance_ids.txt")
parser.add_argument("--out", default="results/window_b/permutation.json")
args = parser.parse_args()

text = Path(args.ids).read_text()
ids = text.split()
order = permutation(ids, SEED)
Path(args.out).write_text(
    json.dumps(
        {
            "seed": SEED,
            "method": "sort ids lexicographically, random.Random(seed).shuffle",
            "ids_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "order": order,
        },
        indent=1,
    )
    + "\n"
)
print(len(order), order[:3])
