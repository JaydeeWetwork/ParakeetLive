# Parakeet environment: keep all caches on the WSL ext4 disk under /opt/parakeet
export PARAKEET_HOME=/opt/parakeet
export HF_HOME=/opt/parakeet/hf-cache
export HF_HUB_CACHE=/opt/parakeet/hf-cache/hub
export NEMO_CACHE_DIR=/opt/parakeet/models/nemo-cache
export TORCH_HOME=/opt/parakeet/models/torch
export XDG_CACHE_HOME=/opt/parakeet/models/xdg-cache
export NUMBA_CACHE_DIR=/opt/parakeet/models/numba-cache
export TOKENIZERS_PARALLELISM=false
