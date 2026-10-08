#!/opt/parakeet/venv/bin/python
"""One-time: build /opt/parakeet/models/parakeet-tdt-0.6b-v2-fast from the .nemo (log 44).
Extracts config + tokenizer artifacts (everything except the fp32 pickle) and writes the state dict as
safetensors with encoder tensors cast to bfloat16. Re-run after replacing the .nemo."""
import hashlib, json, os, sys, tarfile, time
sys.path.insert(0, "/opt/parakeet/scripts")
import transcribe as tx  # noqa: F401  (cache env vars)
import plive_fastload as fl
import torch
import nemo.collections.asr as nemo_asr
import safetensors.torch as st

t0 = time.perf_counter()
os.makedirs(fl.FAST_DIR, exist_ok=True)
man_path = os.path.join(fl.FAST_DIR, "manifest.json")
if os.path.exists(man_path):
    os.remove(man_path)                      # incomplete until the end
src = os.path.realpath(fl.NEMO_FILE)
mode = "r:gz" if open(src, "rb").read(2) == b"\x1f\x8b" else "r:"
with tarfile.open(src, mode) as tf:
    members = [m for m in tf.getmembers() if not m.name.endswith("model_weights.ckpt")]
    names = [m.name for m in members]
    tf.extractall(fl.FAST_DIR, members=members, filter="data")
print("extracted:", names, "compression:", mode)
model = nemo_asr.models.ASRModel.restore_from(fl.NEMO_FILE, map_location="cpu")
sd = model.state_dict()
out, seen, shared, cast = {}, {}, 0, 0
for k, v in sd.items():
    v = v.detach()
    if k.startswith("encoder.") and v.is_floating_point():
        v = v.to(torch.bfloat16); cast += 1
    key = (v.untyped_storage().data_ptr(), v.storage_offset(), tuple(v.shape))
    if key in seen:
        shared += 1
    seen[key] = k
    out[k] = v.clone().contiguous()
wpath = os.path.join(fl.FAST_DIR, fl.WEIGHTS)
st.save_file(out, wpath, metadata={"source": os.path.basename(src), "encoder_dtype": "bfloat16"})
h = hashlib.sha256()
with open(src, "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""):
        h.update(b)
man = {"complete": True, "source": src, "source_sha256": h.hexdigest(), "tensors": len(out),
       "encoder_tensors_bf16": cast, "shared_tensors": shared, "weights_bytes": os.path.getsize(wpath),
       "class": type(model).__name__, "built": time.strftime("%Y-%m-%d %H:%M:%S"),
       "torch": torch.__version__, "build_s": round(time.perf_counter() - t0, 1)}
with open(man_path, "w") as f:
    json.dump(man, f, indent=1)
print(json.dumps(man, indent=1))
