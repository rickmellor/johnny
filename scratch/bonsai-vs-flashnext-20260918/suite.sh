#!/usr/bin/bash
# suite.sh <name> <port> <model> <concurrency>
cd ~/repos/johnny/scratch/bonsai-vs-flashnext-20260918; N=$1 PORT=$2 M=$3 C=$4; mkdir -p $N
python3 run60.py $N http://localhost:$PORT $M > $N/run60.log 2>&1
/home/rick/.local/share/pipx/venvs/johnny-fleet/bin/python /home/rick/repos/johnny/src/johnny/scripts/arc_eval.py --base-url http://localhost:$PORT/v1 --model $M --limit 200 --concurrency $C --max-tokens 2048 --timeout 600 --disable-thinking --out $N/arc.jsonl > $N/arc.log 2>&1
HF_ALLOW_CODE_EVAL=1 /home/rick/.local/share/pipx/venvs/johnny-fleet/bin/python -m lm_eval run --model local-chat-completions --model_args "base_url=http://127.0.0.1:$PORT/v1/chat/completions,model=$M,num_concurrent=$C,max_retries=3,tokenized_requests=False,timeout=600" --tasks humaneval --apply_chat_template --log_samples --output_path $N/humaneval --confirm_run_unsafe_code --gen_kwargs "max_gen_toks=2048" "until=[]" "chat_template_kwargs={'enable_thinking': False}" > $N/humaneval.log 2>&1
/home/rick/.local/share/pipx/venvs/johnny-fleet/bin/python /home/rick/repos/johnny/src/johnny/scripts/humaneval_chat_score.py $(ls -t $N/humaneval/**/samples_humaneval_*.jsonl 2>/dev/null | head -1) > $N/humaneval-score.log 2>&1
echo SUITEDONE
