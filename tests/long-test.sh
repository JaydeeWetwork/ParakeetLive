#!/bin/bash
# Long-audio (~12 min) VRAM test: exercises local attention path (run as root inside WSL)
set -euo pipefail
source /opt/parakeet/scripts/env.sh
S=/opt/parakeet/samples; cd $S
[ -f long-12min.mp3 ] || { for i in $(seq 1 48); do echo "file '$S/2086-149220-0033.wav'"; echo "file '$S/espeak-test.wav'"; done > /tmp/list.txt
  ffmpeg -nostdin -loglevel error -y -f concat -safe 0 -i /tmp/list.txt -ar 16000 -ac 1 -c:a libmp3lame -q:a 5 long-12min.mp3; }
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) > /tmp/vram-long.txt 2>/dev/null &
MON=$!
time /opt/parakeet/venv/bin/python /opt/parakeet/scripts/transcribe.py $S/espeak-test.wav $S/long-12min.mp3 -o $S/out "$@" 2> >(grep -v '^\[NeMo W' >&2)
kill $MON || true
echo "baseline_vram_mib=$(head -1 /tmp/vram-long.txt) peak_gpu_vram_mib=$(sort -n /tmp/vram-long.txt | tail -1)"
wc -w $S/out/long-12min.txt; head -c 400 $S/out/long-12min.txt; echo; tail -8 $S/out/long-12min.srt
echo LONG_DONE
