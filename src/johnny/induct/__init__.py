"""Model induction (§3.6) — find a model's optimal placement on this hardware.

Default = tuning only (make it run well). Quality benchmarking is opt-in (--bench).
A resumable state machine; a seeded search (not a brute grid) with KV-preflight
pruning so impossible contexts cost nothing. Reuses the existing tuning scripts
(bench.sh / wait-ready.sh) and the vLLM driver rather than reimplementing them.
"""
