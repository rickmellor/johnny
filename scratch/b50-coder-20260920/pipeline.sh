#!/usr/bin/bash
cd ~/repos/johnny/scratch/b50-coder-20260920
until grep -q "pass@1" qwen3coder-udq3kxl/humaneval-score.log 2>/dev/null; do sleep 20; done
echo "HE qwen3coder-udq3kxl: $(grep pass@1 qwen3coder-udq3kxl/humaneval-score.log)"
python3 runhard.py qwen3coder-udq3kxl
run() { # name model think-too
  ./serve.sh $2 32768 4 || { echo "$1 failed to serve"; return; }
  echo "HE $1: $(./humaneval.sh $1)"; python3 runhard.py $1
  [ "$3" = think ] && python3 runhard.py $1-think think
}
run qwen3coder-q3km unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q3_K_M.gguf
run qwen36-35b-udiq3s unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-IQ3_S.gguf think
run qwen36-35b-q2k bartowski/Qwen_Qwen3.6-35B-A3B-GGUF/Qwen_Qwen3.6-35B-A3B-Q2_K.gguf
echo PIPEDONE
