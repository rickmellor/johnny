"""FP8-Dynamic quantization with llm-compressor, matching RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic:
weights FP8 E4M3 per-channel (static), activations FP8 per-token (dynamic), targets Linear,
ignoring the MoE routers, the vision tower and lm_head. No calibration data is needed for this scheme.

Usage: python fp8_dynamic.py <src_dir> <dst_dir> [device]   (device: cpu | cuda)
"""
import sys, time, json, torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoConfig
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

src, dst = sys.argv[1], sys.argv[2]
device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
t0 = time.time()
cfg = AutoConfig.from_pretrained(src)
print("arch", cfg.architectures, flush=True)
model = AutoModelForImageTextToText.from_pretrained(src, dtype=torch.bfloat16, device_map=device, trust_remote_code=True)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
try:
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
except Exception as e:
    print("processor load failed, falling back to tokenizer:", e); from transformers import AutoTokenizer; processor = AutoTokenizer.from_pretrained(src)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["lm_head", "re:.*router.*", "re:.*vision_tower.*", "re:.*embed_vision.*", "re:.*audio_tower.*", "re:.*embed_audio.*"],
)
oneshot(model=model, recipe=recipe)
print(f"quantized in {time.time()-t0:.0f}s", flush=True)
model.save_pretrained(dst, save_compressed=True)
processor.save_pretrained(dst)
print(f"saved to {dst} in {time.time()-t0:.0f}s", flush=True)
q = json.load(open(f"{dst}/config.json")).get("quantization_config", {})
print("quantization_config:", json.dumps({k: v for k, v in q.items() if k != "ignore"}, indent=1)[:800])
print("ignored modules:", len(q.get("ignore", [])))
