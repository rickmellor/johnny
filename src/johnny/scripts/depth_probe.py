#!/usr/bin/env python3
"""
Context-depth probe: prefill tok/s and decode tok/s at chosen prompt depths.

Decode speed *at depth* is what decides whether a seat has usable context, and
it is not predictable from a short-prompt benchmark: one model here fell from
49 tok/s to 8 tok/s by 30K tokens of context while another stayed flat. This
probe measures it directly against ANY OpenAI-compatible endpoint (vLLM or
llama.cpp): for each requested depth it builds a long synthetic code document
(small Python functions, the numbers change from block to block), asks a short
question about it, and streams the answer.

Method
  - There is no tokenizer here, so the document is sized at ~75 tokens per
    function block and the REAL prompt size is taken from the response's
    `usage.prompt_tokens` (`stream_options.include_usage`).
  - Every prompt starts with a random header line and the request carries
    `"cache_prompt": false` (llama.cpp), so a prefix cache cannot fake the
    prefill number.
  - TTFT = time to the first content delta.
      prefill tok/s = prompt_tokens / TTFT
      decode  tok/s = (completion_tokens - 1) / (total - TTFT)
  - HTTP 400 for a depth (context too small) is recorded as an error entry and
    the probe continues with the next depth; the exit code stays 0.

CAVEAT: TTFT includes any time the request spent queued behind other work, so
the prefill figure is only meaningful against an IDLE seat (and decode tok/s is
the single-stream rate only when nothing else is batched alongside). Run it
against an idle seat.

Usage:
  python3 depth_probe.py --base-url http://127.0.0.1:8124/v1 --model coder \\
      [--depths 2000,16000,32000] [--max-tokens 64] [--thinking] \\
      [--timeout 3600] [--api-key EMPTY] [--out result.json]

Output: one human-readable line per depth, then a final machine line
  DEPTHPROBE_RESULT {"points": [...]}
with one object per depth:
  {"target_tokens", "prompt_tokens", "ttft_s", "prefill_tok_s", "decode_tok_s",
   "completion_tokens"}   or   {"target_tokens", "error"}
"""
import argparse
import json
import secrets
import sys
import time
import urllib.error
import urllib.request

TOKENS_PER_BLOCK = 75

_BLOCK = (
    "def process_record_{i}(rec):\n"
    "    # validate record batch {i}, normalise its fields and return (id, name, total)\n"
    "    if not rec or 'id' not in rec: raise ValueError('bad record {i}')\n"
    "    total = sum(float(x) * {k} for x in rec.get('items', []))\n"
    "    return rec['id'], rec.get('name', '').strip().title(), round(total, 2)\n\n"
)

_QUESTION = "In two sentences, what do these functions do and what would you refactor?"


def build_prompt(target_tokens: int) -> str:
    """A unique prompt of roughly `target_tokens` tokens (random first line)."""
    blocks = max(1, target_tokens // TOKENS_PER_BLOCK)
    doc = "".join(_BLOCK.format(i=i, k=(i * 7) % 97 + 1) for i in range(blocks))
    header = f"# probe {secrets.token_hex(8)}\n"
    return header + "Here is a module.\n\n" + doc + "\n" + _QUESTION


def probe(base_url: str, model: str, target_tokens: int, max_tokens: int = 64,
          thinking: bool = False, timeout: float = 3600, api_key: str = "EMPTY") -> dict:
    """One streamed request at one depth -> a result point (or an error point)."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": build_prompt(target_tokens)}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "cache_prompt": False,
    }
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    point: dict = {"target_tokens": target_tokens}
    t0 = time.perf_counter()
    t_first = None
    usage = None
    pieces = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    # with --thinking the first generated token arrives as reasoning
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        pieces += 1
                        if t_first is None:
                            t_first = time.perf_counter()
        t_end = time.perf_counter()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace").strip().replace("\n", " ")[:300]
        point["error"] = f"HTTP {e.code}: {detail or e.reason}"
        return point
    except Exception as e:  # timeout, connection reset, ...
        point["error"] = f"{type(e).__name__}: {e}"[:300]
        return point

    if t_first is None:
        point["error"] = "no content received"
        return point
    if not usage or not usage.get("prompt_tokens"):
        point["error"] = "server sent no usage chunk (stream_options.include_usage unsupported?)"
        return point
    ttft = t_first - t0
    prompt_tokens = int(usage["prompt_tokens"])
    completion_tokens = int(usage.get("completion_tokens") or pieces)
    decode_s = t_end - t_first
    point.update({
        "prompt_tokens": prompt_tokens,
        "ttft_s": round(ttft, 3),
        "prefill_tok_s": round(prompt_tokens / ttft, 1) if ttft > 0 else None,
        "decode_tok_s": round((completion_tokens - 1) / decode_s, 2)
        if completion_tokens > 1 and decode_s > 0 else None,
        "completion_tokens": completion_tokens,
    })
    return point


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Prefill/decode tok/s at chosen prompt depths (OpenAI-compatible endpoint). "
                    "TTFT includes queueing: run against an idle seat.")
    ap.add_argument("--base-url", required=True, help="e.g. http://127.0.0.1:8124/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--depths", default="2000,16000,32000",
                    help="comma list of approximate prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--thinking", action="store_true",
                    help="leave thinking on (default sends chat_template_kwargs enable_thinking=false)")
    ap.add_argument("--timeout", type=float, default=3600, help="per-request seconds")
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--out", help="also write the result JSON here")
    args = ap.parse_args(argv)

    try:
        depths = [int(x) for x in args.depths.split(",") if x.strip()]
    except ValueError:
        ap.error(f"--depths must be a comma list of integers, got {args.depths!r}")
    if not depths:
        ap.error("--depths is empty")

    points = []
    for depth in depths:
        p = probe(args.base_url, args.model, depth, args.max_tokens, args.thinking,
                  args.timeout, args.api_key)
        points.append(p)
        if "error" in p:
            print(f"depth ~{depth:>7d}: ERROR {p['error']}", flush=True)
        else:
            dec = p["decode_tok_s"]
            print(f"depth ~{depth:>7d}: prompt {p['prompt_tokens']:>7d} tok | "
                  f"TTFT {p['ttft_s']:8.2f} s | prefill {p['prefill_tok_s']:9.1f} tok/s | "
                  f"decode {dec if dec is not None else float('nan'):7.2f} tok/s "
                  f"({p['completion_tokens']} tok)", flush=True)

    result = {"points": points}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    print("DEPTHPROBE_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
