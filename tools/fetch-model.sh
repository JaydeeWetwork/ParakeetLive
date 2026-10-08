#!/bin/bash
# Downloads parakeet-tdt-0.6b-v2 into /opt/parakeet (run as root inside WSL)
set -euo pipefail
source /opt/parakeet/scripts/env.sh
/opt/parakeet/venv/bin/python - <<'PY'
import os, time
import nemo.collections.asr as nemo_asr
t=time.time()
m = nemo_asr.models.ASRModel.from_pretrained("nvidia/parakeet-tdt-0.6b-v2", map_location="cpu")
from huggingface_hub import hf_hub_download
src = hf_hub_download("nvidia/parakeet-tdt-0.6b-v2", "parakeet-tdt-0.6b-v2.nemo")  # already cached, no re-download
out = "/opt/parakeet/models/parakeet-tdt-0.6b-v2.nemo"
if os.path.lexists(out): os.remove(out)
os.symlink(os.path.realpath(src), out)
print("linked", out, "->", os.path.realpath(src), os.path.getsize(out)//2**20, "MiB; total", round(time.time()-t,1), "s")
PY
du -sh /opt/parakeet/* ; ls -la /root/.cache 2>/dev/null || true
echo MODEL_DONE
