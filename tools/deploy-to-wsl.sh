#!/bin/bash
# Deploy the repo's WSL-side code into /opt/parakeet/scripts (run as root inside WSL):
#   wsl -d Ubuntu-24.04 -u root -e bash tools/deploy-to-wsl.sh     (in PowerShell, from the repo folder)
# Copies server/live_server.py, server/plive_fastload.py, batch/transcribe.py, batch/env.sh (CRLF stripped).
# A file that changed is backed up first as <name>.bak-<timestamp> next to it. Nothing is deleted.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
DST=/opt/parakeet/scripts
ts=$(date +%Y%m%d-%H%M%S)
mkdir -p "$DST"
for src in server/live_server.py server/plive_fastload.py batch/transcribe.py batch/env.sh; do
  f=$(basename "$src")
  if [ -f "$DST/$f" ] && cmp -s <(tr -d '\r' < "$REPO/$src") "$DST/$f"; then
    echo "unchanged  $f"; continue
  fi
  [ -f "$DST/$f" ] && cp -p "$DST/$f" "$DST/$f.bak-$ts"
  tr -d '\r' < "$REPO/$src" > "$DST/$f"
  echo "deployed   $f"
done
chmod +x "$DST/live_server.py" "$DST/transcribe.py"
ln -sfn "$DST/transcribe.py" /opt/parakeet/transcribe.py
echo "done $(date -Is) from $REPO"
