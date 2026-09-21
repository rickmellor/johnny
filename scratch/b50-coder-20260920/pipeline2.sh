#!/usr/bin/bash
cd ~/repos/johnny/scratch/b50-coder-20260920; shopt -s globstar
PY=~/.local/share/pipx/venvs/johnny-fleet/bin/python; S=~/repos/johnny/src/johnny/scripts
while pgrep -f "lm_eval run" >/dev/null; do sleep 20; done
$PY $S/humaneval_chat_score.py $(ls -t qwen3coder-udq3kxl/humaneval/**/samples_humaneval_*.jsonl | head -1) > qwen3coder-udq3kxl/humaneval-score.log 2>&1
echo "HE qwen3coder-udq3kxl: $(grep pass@1 qwen3coder-udq3kxl/humaneval-score.log)"
python3 runhard.py qwen3coder-udq3kxl
./serve.sh unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf 16384 4 && python3 runhard.py qwen3coder-q3km
./serve.sh unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf 65536 4 && { python3 runhard.py qwen36-35b-udiq3s; python3 runhard.py qwen36-35b-udiq3s-think think; }
./serve.sh bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q2_K.gguf 65536 4 && python3 runhard.py qwen36-35b-q2k
echo PIPEDONE
