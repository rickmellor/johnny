#!/usr/bin/bash
# serve.sh <model-relpath> [ctx] [np] [extra args] — llama-server on the B50 via Vulkan + Mesa 26 (kisak), port 8124
M=$1; CTX=${2:-32768}; NP=${3:-4}; shift 3 2>/dev/null
docker rm -f b50v >/dev/null 2>&1
docker run -d --name b50v --device /dev/dri --group-add 44 --group-add 992 -p 0.0.0.0:8124:8080 -v $HOME/repos/llama.cpp-prism/build-vk/bin:/bin-p:ro -v /mnt/data/models:/models:ro -e LD_LIBRARY_PATH=/bin-p prism-vk-mesa-new bash -c "I=\$(/bin-p/llama-bench --list-devices 2>&1 | grep -i 'Arc' | grep -o 'Vulkan[0-9]*' | tr -d 'Vulkan'); export GGML_VK_VISIBLE_DEVICES=\$I; exec /bin-p/llama-server -m /models/$M -ngl 99 -fa on -c $CTX -np $NP --host 0.0.0.0 --port 8080 --jinja --alias coder $*" >/dev/null
for i in $(seq 1 120); do curl -sf localhost:8124/health >/dev/null && { echo "up: $M"; exit 0; }; [ "$(docker inspect -f '{{.State.Running}}' b50v)" = true ] || { echo "DIED"; docker logs b50v 2>&1 | grep -i -E "error|failed|out of" | tail -3 | cut -c1-200; exit 1; }; sleep 3; done
