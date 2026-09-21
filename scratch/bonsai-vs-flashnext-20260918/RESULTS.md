# Ternary Bonsai 2 27B (RTX 4080, llama.cpp fork) vs Qwen3.8-Flash-Next-AWQ (4× R9700, vLLM) — 2026-09-18

22 prompts, temperature 0, thinking off on both, max 1024 tokens. Code = executed against hidden tests; numeric = exact;
knowledge/format = checked by hand (Claude graded). Prompts in prompts.py, raw answers in answers-*.json.

| category | Bonsai | Flash-Next | notes |
|---|---|---|---|
| math (5) | 4 | 5 | Bonsai m4: correct reasoning but dithered and hit the 1024-token cap before answering |
| code (5) | 5 | 5 | all tests pass on both |
| knowledge (5) | 2.5 | 4 | Bonsai k1 said Platinum/Pt for Z=74 (tungsten); k3 claimed `-f` adds "mount options" (half credit). BOTH missed k2 (Voyager 1's second flyby = Saturn; both said Jupiter) |
| instruction (3) | 2 | 3 | Bonsai i2 haiku was 4/5/4 syllables and it reported 21 |
| extraction (2) | 2 | 2 | e2: Bonsai's summary was slightly more faithful |
| logic (2) | 2 | 2 | |
| **total** | **17.5 / 22 (80 %)** | **21 / 22 (95 %)** | |

Speed during the run: Bonsai ~70 tok/s single-stream; Flash-Next ~40 tok/s (MTP on) and ~30 % more verbose on math.
Bonsai answered every prompt in ≤15 s; Flash-Next's long math answers took up to 16 s.

---
# Round 2 (2026-09-18, later): 424 items — ARC-Challenge 200 + HumanEval 164 + 60-prompt hand set

Same settings (temperature 0, thinking off). Bonsai on the RTX 4080 single slot (concurrency 1); Flash-Next on 4× R9700 (concurrency 8).
Hand-set rubric items graded by Claude; raw answers in answers60-*.json, ARC samples in */arc.jsonl, HumanEval samples under */humaneval/.

| suite | Bonsai | Flash-Next | discordant items |
|---|---|---|---|
| ARC-Challenge (200) | 192 = 96.0 % | 194 = 97.0 % | 4 (Bonsai-only wrong 3, Flash-only wrong 1, both wrong 5) |
| HumanEval (164) | 151 = 92.1 % | 156 = 95.1 % | 12 (6 each way; both fail decode_cyclic, decode_shift) |
| hand set: knowledge (15) | 13 | 15 | Bonsai: PCIe lane = "one differential pair"; RAID-5's 5 = "minimum drives" |
| hand set: instruction (15) | 13 | 12 | both: word-order sentence, reversing "benchmark" (kramhcneb; B "mkcenreb", F "kramhcreb"); Flash also blew the ≤6-word bullets |
| hand set: extraction (15) | 11 | 13 | both: 3×4.50+2×12.25 (B 33.75, F 39); "third word of second sentence" (both "replaced"); Bonsai also miscounted hostnames (3/4) and a sum (355/350) |
| hand set: reasoning (15) | 15 | 15 | |
| **all 424** | **395 = 93.2 %** | **405 = 95.5 %** | |
| + round-1 22-set | 17.5 / 22 | 21 / 22 | |

Knowledge across both rounds: Bonsai 15.5/20, Flash-Next 19/20. Reasoning + code + logic: near parity (ARC −1 pt, HumanEval −3 pts, hand reasoning equal).
Wall time: Bonsai ~45 min on one slot; Flash-Next ~25 min.
