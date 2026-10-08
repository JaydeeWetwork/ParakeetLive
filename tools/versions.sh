#!/bin/bash
source /opt/parakeet/scripts/env.sh
echo "date: $(date -Is)"
grep PRETTY_NAME /etc/os-release; uname -r
/opt/parakeet/venv/bin/python --version
/opt/parakeet/venv/bin/python - <<'PY' 2>/dev/null
import torch, importlib.metadata as md
print("torch", torch.__version__, "| CUDA (torch build)", torch.version.cuda, "| cuDNN", torch.backends.cudnn.version(), "| GPU", torch.cuda.get_device_name(0), "| bf16", torch.cuda.is_bf16_supported())
for p in ["nemo_toolkit","torchaudio","lightning","huggingface_hub","numpy","lhotse"]:
    try: print(p, md.version(p))
    except Exception as e: print(p, "n/a")
PY
ffmpeg -version | head -1
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "--- /opt/parakeet"; du -sh /opt/parakeet/* /opt/parakeet/models/* 2>/dev/null
ls -la /opt/parakeet/models
echo "--- /root/.cache"; du -sh /root/.cache/* 2>/dev/null
df -h /
