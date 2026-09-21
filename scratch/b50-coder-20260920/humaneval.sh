#!/usr/bin/bash
# humaneval.sh <name> [think 0|1]
cd ~/repos/johnny/scratch/b50-coder-20260920; N=$1; PY=~/.local/share/pipx/venvs/johnny-fleet/bin/python; S=~/repos/johnny/src/johnny/scripts; mkdir -p $N; shopt -s globstar
HF_ALLOW_CODE_EVAL=1 $PY -m lm_eval run --model local-chat-completions --model_args "base_url=http://127.0.0.1:8124/v1/chat/completions,model=coder,num_concurrent=4,max_retries=3,tokenized_requests=False,timeout=900" --tasks humaneval --apply_chat_template --log_samples --output_path $N/humaneval --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=2048" "until=[]" "chat_template_kwargs={'enable_thinking': False}" > $N/humaneval.log 2>&1
$PY $S/humaneval_chat_score.py $(ls -t $N/humaneval/**/samples_humaneval_*.jsonl | head -1) > $N/humaneval-score.log 2>&1; grep -E "pass@1" $N/humaneval-score.log
