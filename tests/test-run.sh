#!/bin/bash
# GPU smoke test for the Parakeet install (run as root inside WSL)
set -euo pipefail
source /opt/parakeet/scripts/env.sh
S=/opt/parakeet/samples; mkdir -p $S
cd $S
[ -f 2086-149220-0033.wav ] || wget -q https://dldata-public.s3.us-east-2.amazonaws.com/2086-149220-0033.wav
espeak-ng -s 150 -w espeak-test.wav "Hello Jaydee. This is a local transcription test running on the workstation graphics card. The bundle was a win."
ffmpeg -nostdin -loglevel error -y -i 2086-149220-0033.wav -c:a libmp3lame -q:a 4 librispeech-sample.mp3
du -sh /opt/parakeet/models/* || true
# sample nvidia-smi every 200 ms for peak VRAM (whole GPU, includes ~0.4 GB Windows desktop)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done ) > /tmp/vram.txt 2>/dev/null &
MON=$!
time /opt/parakeet/venv/bin/python /opt/parakeet/scripts/transcribe.py \
   $S/2086-149220-0033.wav $S/espeak-test.wav $S/librispeech-sample.mp3 -o $S/out --json "$@" 2> >(grep -v '^\[NeMo W' >&2)
kill $MON || true
echo "baseline_vram_mib=$(head -1 /tmp/vram.txt) peak_gpu_vram_mib=$(sort -n /tmp/vram.txt | tail -1)"
for f in $S/out/*.txt; do echo "== $f"; cat "$f"; done
echo "== srt"; cat $S/out/2086-149220-0033.srt
echo TEST_DONE
