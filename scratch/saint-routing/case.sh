#!/bin/bash
# usage: case.sh <tag> <model> <prompt>
set -a; . ~/.config/saint/env; set +a
tag="$1"; model="$2"; prompt="$3"
out=~/rt/$tag
python3 - "$model" "$prompt" > $out.req.json <<'EOF'
import json,sys
print(json.dumps({"model":sys.argv[1],"max_tokens":64,
 "messages":[{"role":"user","content":sys.argv[2]}]}))
EOF
echo "=== $tag  model=$model ==="
curl -s -D $out.hdr -o $out.body -w 'HTTP=%{http_code} TIME=%{time_total}s\n' \
  -m 300 -X POST http://127.0.0.1:4000/v1/chat/completions \
  -H 'Content-Type: application/json' --data-binary @$out.req.json
echo "--- x-saint headers ---"
grep -i -E '^(HTTP/|x-saint)' $out.hdr
echo "--- body (head) ---"
head -c 700 $out.body
echo
echo "--- saint log show --limit 2 ---"
saint log show --limit 2 2>&1 | head -10
