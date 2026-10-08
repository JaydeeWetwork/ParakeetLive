#!/opt/parakeet/venv/bin/python
"""Verify the fast loader against the stock .nemo path (log 44): every parameter and buffer bit-identical
after the same encoder->bf16 cast, same model class/config, and same transcripts on the test samples
(fp32 CPU-free check on GPU with deterministic settings)."""
import glob, json, os, sys, time, wave
if len(sys.argv) < 2 or not os.path.isdir(sys.argv[1]):
    sys.exit("usage: verify-fast-load.py <folder with 16 kHz mono 16-bit .wav clips>")
WAVS = sorted(glob.glob(os.path.join(sys.argv[1], "*.wav")))
sys.path.insert(0, "/opt/parakeet/scripts")
import transcribe as tx
import plive_fastload as fl
import numpy as np
import torch
import nemo.collections.asr as nemo_asr

t = time.perf_counter(); a = nemo_asr.models.ASRModel.restore_from(str(tx.MODEL_FILE), map_location="cpu"); a.eval(); a.encoder.to(torch.bfloat16)
print(f"stock restore_from: {time.perf_counter()-t:.1f} s")
t = time.perf_counter(); b = fl.load_cpu(); print(f"fast load_cpu: {time.perf_counter()-t:.1f} s")
print("class", type(a).__name__, type(b).__name__)
def tensors(m):
    d = {"P:" + k: v for k, v in m.named_parameters()}
    d.update({"B:" + k: v for k, v in m.named_buffers()})
    return d
ta, tb = tensors(a), tensors(b)
assert ta.keys() == tb.keys(), set(ta) ^ set(tb)
bad = [k for k in ta if ta[k].dtype != tb[k].dtype or ta[k].shape != tb[k].shape or not torch.equal(ta[k], tb[k])]
print(f"params+buffers compared: {len(ta)}  mismatches: {len(bad)} {bad[:5]}")
print("config equal:", a.cfg == b.cfg)
res = {}
for name, m in (("stock", a), ("fast", b)):
    tx.disable_cuda_graph_decoder(m)
    m.preprocessor.featurizer.dither = 0.0; m.preprocessor.featurizer.pad_to = 0
    m.to("cuda")
    outs = {}
    for f in WAVS:
        with wave.open(f) as w:
            x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype("float32") / 32768
        sig = torch.from_numpy(x).unsqueeze(0).cuda(); ln = torch.tensor([sig.shape[1]], device="cuda")
        with torch.inference_mode():
            feats, flen = m.preprocessor(input_signal=sig, length=ln)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                enc, elen = m.encoder(audio_signal=feats, length=flen)
            hyp = m.decoding.rnnt_decoder_predictions_tensor(encoder_output=enc.float(), encoded_lengths=elen)
        hyp = hyp[0] if isinstance(hyp, tuple) else hyp
        outs[f] = hyp[0].text
    res[name] = outs
    m.cpu(); torch.cuda.empty_cache()
print("TRANSCRIPTS", json.dumps(res, indent=1))
print("transcripts identical:", res["stock"] == res["fast"])
print("RESULT", "PASS" if not bad and res["stock"] == res["fast"] else "CHECK")
