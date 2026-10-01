import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import record_loom_fixtures as rec

UNMASKED = [
    r"D:\models\gguf\Qwen3-8B-Q4_K_M.gguf",
    r"C:\Users\someone\models\m.gguf",
    "/home/someone/models/m.gguf",
    "/mnt/data/models/m.gguf",
    "/Users/someone/lib/libllama.dylib",
    "/opt/llama/build/bin/libllama.so",
    r"\\fileserver\share\m.gguf",
    "http://10.1.2.3:8020/healthz",
    "server at 192.168.0.5",
]


def test_paths_and_addresses_are_masked_in_nested_payloads():
    for value in UNMASKED:
        masked = rec.mask({"a": [{"b": f"loaded {value} ok"}], "c": value})
        text = json.dumps(masked)
        assert value not in text, value
        assert "<path>" in text or "<host>" in text


def test_masking_is_stable_and_leaves_ordinary_text_alone():
    payload = {"msg": "Work in /work, read main.py", "n": 3, "ok": True, "x": None}
    assert rec.mask(payload) == payload
    once = rec.mask({"p": "/home/u/m.gguf"})
    assert rec.mask(once) == once


def test_distinct_request_ids_stay_distinct_and_repeats_map_the_same():
    rec.reset_request_ids()
    a, b = "chatcmpl-0123456789abcdef", "chatcmpl-fedcba9876543210"
    assert rec.mask_text(a) == "chatcmpl-000001"
    assert rec.mask_text(b) == "chatcmpl-000002"
    assert rec.mask_text(f'data: {{"id": "{a}"}}') == 'data: {"id": "chatcmpl-000001"}'


def test_header_and_sse_of_one_recording_keep_matching_ids(tmp_path):
    rec.reset_request_ids()
    a, b = "chatcmpl-0123456789abcdef", "chatcmpl-fedcba9876543210"
    rec._write(tmp_path, "t1.sse", f'data: {{"id": "{a}"}}\n\n')
    rec._write(tmp_path, "t1.headers.json", {"x-evoke-request-id": a})
    rec._write(tmp_path, "t2.sse", f'data: {{"id": "{b}"}}\n\n')
    rec._write(tmp_path, "t2.headers.json", {"x-evoke-request-id": b})
    h1 = json.loads((tmp_path / "t1.headers.json").read_text())["x-evoke-request-id"]
    h2 = json.loads((tmp_path / "t2.headers.json").read_text())["x-evoke-request-id"]
    assert h1 in (tmp_path / "t1.sse").read_text()
    assert h2 in (tmp_path / "t2.sse").read_text()
    assert h1 != h2
    assert h1 not in (tmp_path / "t2.sse").read_text()


def test_configured_host_is_masked_even_when_it_is_a_name():
    assert rec.mask_text("see http://evoke-box:8020/x", hosts=("evoke-box",)) == (
        "see http://<host>:8020/x"
    )


def test_write_masks_json_and_raw_text(tmp_path):
    rec.set_masked_hosts(())
    rec._write(tmp_path, "a.json", {"model_path": "/mnt/data/m.gguf"})
    rec._write(tmp_path, "b.sse", "data: /home/u/m.gguf\n\n")
    assert "/mnt/data" not in (tmp_path / "a.json").read_text()
    assert "/home/u" not in (tmp_path / "b.sse").read_text()


def test_header_values_are_masked(tmp_path):
    class Resp:
        status_code = 200
        headers = {"x-evoke-request-id": "chatcmpl-0123456789abcdef", "content-type": "a"}

    rec._write(tmp_path, "h.json", rec._record(Resp(), {}))
    assert "0123456789abcdef" not in (tmp_path / "h.json").read_text()
