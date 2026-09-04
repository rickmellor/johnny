# Qwen3.5-122B-A10B AWQ — context expansion + 1M YaRN trial — 2026-08-27

Registry `qwen-122b-awq` (cyankiwi AWQ-4bit W4A16, 75 GiB, TP4). Config: 48 layers = 36 GDN + 12 full attention (2 KV heads × 256) → KV 24 KB/token bf16 / 12 KB fp8; GDN state 147 MB/seq; native 262,144.
The old 131K seat was capped by max_num_seqs 128 (128 × 147 MB GDN state), not by KV.

| placement | image / KV | ctx | perf (warmed) | KV pool | needle | notes |
|---|---|---|---|---|---|---|
| qwen-122b-awq-mtp (old) | v0.20.2 bf16 | 131K, 128 slots | 82.6 single (UN-warmed record) | — | ctxsafe 90K (08-06) | superseded |
| **qwen-122b-awq-mtp-262k-seqs16** | v0.20.2 bf16 | 262K, 16 slots | **114.1 single / 234 agg@4** | 440K tok (1.69×@262K) | 6/6 single to 256K + 4×256K (serialized) | **profile `qwen122b-tp4`** |
| v0280-aiter-kvfp8-mtp-262k-seqs16 | v0.28.0 fp8+AITER | 262K, 16 slots | 121.5 single / 176 agg@4; MTP acc 88.6%; 256K TTFT 68 s (vs 218) | 710K tok (2.71×) | single 6/6; **concurrent 256K: 2/4 garbage, 2×256K: 1 empty** (0 preemptions) | NOT behind SAINT until isolated (AITER vs MTP under batching) |
| **v0280-aiter-kvfp8-mtp-1m-yarn4-seqs4** | v0.28.0 fp8+AITER, YaRN ×4 | **1,048,576**, 4 slots | short 121; **22.1 t/s @300K; 8.3 t/s @1.04M** | 1.10M tok (1.05×@1M) | **5/5 to 1.04M** (25%/75% positions) | **profile `qwen122b-1m`**; prefill 2190/2010/1365/956 t/s @300K/500K/750K/1M; cold 1M TTFT ~18 min, 8.7 s cached |

YaRN: `--hf-overrides '{"text_config":{"rope_parameters":{rope_type yarn, factor 4, original_max_position_embeddings 262144, + existing theta/partial_rotary/mrope fields},"max_position_embeddings":1048576}}'` — must be nested under text_config for this multimodal config (top-level override is silently ignored → vLLM refuses max_model_len > 262144). Static YaRN degrades short prompts slightly → separate profile only.
Decode collapses with depth on the fp8+AITER path (121 → 22 @300K → 8 @1M): far below the KV-bandwidth bound (~6 ms/token @300K) → AITER fp8-KV decode kernel scaling / MTP verify at depth; A/B vs bf16/no-AITER at 300K is the next diagnostic.
Profiles: `qwen122b-tp4` (daily candidate; 114 t/s, chat+coder aliased) and `qwen122b-1m` (massive-context, single user). SAINT resolves chat/coder via johnny resolve.
Open: (1) concurrent-deep corruption isolation on the 0.28 fp8+AITER 262K seat; (2) decode-at-depth A/B; (3) AutomationBench vs Qwen3.8-27B effort-low (40%) before making qwen122b-tp4 the boot default; (4) johnny: warm GDN seats before induct/bench perf (stale 82.6-type records).
