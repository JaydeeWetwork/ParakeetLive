#!/opt/parakeet/venv/bin/python
"""Local English transcription with NVIDIA Parakeet TDT 0.6B v2 (runs inside WSL Ubuntu-24.04).

Usage:  transcribe.py INPUT [INPUT ...] [-o OUTDIR] [--chunk-min N] [--cpu]  (see --help)
Writes INPUT-stem.txt and INPUT-stem.srt next to each input (or into OUTDIR).
Accepts any audio/video format ffmpeg can read; converts to 16 kHz mono WAV first.
"""
import os

# Keep every cache on the WSL ext4 disk under /opt/parakeet (never /root/.cache or /mnt/*).
_BASE = "/opt/parakeet"
os.environ.setdefault("HF_HOME", f"{_BASE}/hf-cache")
os.environ.setdefault("HF_HUB_CACHE", f"{_BASE}/hf-cache/hub")
os.environ.setdefault("NEMO_CACHE_DIR", f"{_BASE}/models/nemo-cache")
os.environ.setdefault("TORCH_HOME", f"{_BASE}/models/torch")
os.environ.setdefault("XDG_CACHE_HOME", f"{_BASE}/models/xdg-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", f"{_BASE}/models/numba-cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import warnings
warnings.filterwarnings("ignore")  # NeMo/Lightning import-time warnings are noise here
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v2"
MODEL_FILE = Path(f"{_BASE}/models/parakeet-tdt-0.6b-v2.nemo")
SR = 16000


def log(msg):
    print(f"[parakeet] {msg}", file=sys.stderr, flush=True)


def probe_duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)], capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def to_wav(src, dst, start=None, length=None):
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += ["-i", str(src)]
    if length is not None:
        cmd += ["-t", f"{length:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)]
    subprocess.run(cmd, check=True)


def srt_time(t):
    t = max(0.0, float(t))
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def disable_cuda_graph_decoder(model):
    from omegaconf import open_dict
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.greedy.use_cuda_graph_decoder = False
    model.change_decoding_strategy(cfg, verbose=False)


def load_model(device, no_cuda_graphs=False):
    import torch
    import nemo.collections.asr as nemo_asr
    logging.getLogger("nemo_logger").setLevel(logging.ERROR)
    if MODEL_FILE.exists():
        model = nemo_asr.models.ASRModel.restore_from(str(MODEL_FILE), map_location="cpu")
    else:
        log(f"local model file missing, downloading {MODEL_ID} into {os.environ['HF_HOME']}")
        model = nemo_asr.models.ASRModel.from_pretrained(MODEL_ID, map_location="cpu")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        model.save_to(str(MODEL_FILE))
    model.eval()
    model._parakeet_long_mode = False
    # Subsampling conv chunking factor 1 = split only when needed (avoids 2**31-element
    # limits on very long audio). NeMo 3.0.0 crashes with factor -1, so never use -1.
    try:
        model.change_subsampling_conv_chunking_factor(1)
    except Exception as e:
        log(f"subsampling chunking unavailable: {e}")
    if no_cuda_graphs:
        disable_cuda_graph_decoder(model)
    if device == "cuda":
        # Half-size weights for the big FastConformer encoder (~1.2 GB instead of ~2.4 GB).
        # Preprocessor, TDT decoder and joint stay fp32 (small; NeMo decodes outside autocast).
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model.encoder.to(dtype)
        model = model.to("cuda")
        return model, dtype
    return model, None


def set_long_audio_mode(model, on, dtype=None):
    """Switch between full attention (best accuracy) and local attention (long audio).
    Only rebuilds the encoder when the mode actually changes."""
    if getattr(model, "_parakeet_long_mode", False) == on:
        return
    if on:
        # Limited-context attention keeps VRAM roughly linear in audio length.
        model.change_attention_model("rel_pos_local_attn", [256, 256])
    else:
        model.change_attention_model("rel_pos")
    if dtype is not None:
        model.encoder.to(dtype)  # attention change rebuilds layers in fp32; cast back
    model._parakeet_long_mode = on


def run_transcribe(model, wav_paths, dtype, device):
    import torch
    ctx = torch.autocast("cuda", dtype=dtype) if device == "cuda" else torch.autocast("cpu", enabled=False)
    with torch.inference_mode(), ctx:
        return model.transcribe([str(p) for p in wav_paths], batch_size=1, timestamps=True, verbose=False)


def segments_from(hyp, offset=0.0):
    segs = []
    ts = getattr(hyp, "timestamp", None) or {}
    for s in ts.get("segment", []) or []:
        text = (s.get("segment") or "").strip()
        if text:
            segs.append((float(s["start"]) + offset, float(s["end"]) + offset, text))
    if not segs and getattr(hyp, "text", "").strip():
        segs.append((offset, offset, hyp.text.strip()))
    return segs


def find_silences(wav, noise_db=-35, min_dur=0.3):
    """Return list of silence midpoints (seconds) using ffmpeg silencedetect."""
    out = subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-i", str(wav), "-af",
                          f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    mids, start = [], None
    for line in out.splitlines():
        if "silence_start:" in line:
            try:
                start = float(line.split("silence_start:")[1].split()[0])
            except ValueError:
                start = None
        elif "silence_end:" in line and start is not None:
            try:
                end = float(line.split("silence_end:")[1].split()[0])
                mids.append((start + end) / 2)
            except ValueError:
                pass
            start = None
    return mids


def plan_chunks(dur, chunk_s, silences, search_s=45.0):
    """Split [0,dur] into pieces <= chunk_s, cutting at the latest silence in the last search_s."""
    cuts, pos = [], 0.0
    while dur - pos > chunk_s:
        target = pos + chunk_s
        cands = [m for m in silences if target - search_s <= m <= target]
        cut = max(cands) if cands else target
        cuts.append(cut)
        pos = cut
    bounds = [0.0] + cuts + [dur]
    return [(bounds[i], bounds[i + 1] - bounds[i]) for i in range(len(bounds) - 1)]


def transcribe_file(model, dtype, device, src, outdir, args):
    import torch
    src = Path(src).resolve()
    tmp = Path(tempfile.mkdtemp(prefix="parakeet-", dir="/tmp"))
    try:
        full = tmp / "full.wav"
        to_wav(src, full)
        dur = probe_duration(full)
        if args.chunk_min > 0:
            chunk_s = args.chunk_min * 60
        elif dur > args.full_attn_max_min * 60:
            chunk_s = args.auto_chunk_min * 60
        else:
            chunk_s = None
        long_mode = args.local_attn == "on" or (args.local_attn == "auto" and dur > args.full_attn_max_min * 60)
        set_long_audio_mode(model, long_mode, dtype)
        if chunk_s and dur > chunk_s:
            plan = plan_chunks(dur, chunk_s, find_silences(full))
        else:
            plan = [(0.0, dur)]
        log(f"{src.name}: {dur/60:.1f} min, attention={'local' if long_mode else 'full'}, pieces={len(plan)}")

        segs, texts = [], []
        for n, (off, length) in enumerate(plan, 1):
            if len(plan) > 1:
                wav = tmp / f"chunk{n:04d}.wav"
                to_wav(full, wav, off, length)
                log(f"  piece {n}/{len(plan)}: {off/60:.1f}-{(off+length)/60:.1f} min")
            else:
                wav = full
            try:
                hyp = run_transcribe(model, [wav], dtype, device)[0]
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                if device == "cuda" and ("CUDA" in str(e) or "out of memory" in str(e)):
                    raise SystemExit(
                        f"[parakeet] GPU ran out of memory on {src.name} ({e.__class__.__name__}: {str(e)[:120]}).\n"
                        f"Close other GPU apps or rerun with smaller pieces, e.g. --chunk-min 3 "
                        f"(PowerShell: -ChunkMin 3).")
                raise
            texts.append(hyp.text.strip())
            segs.extend(segments_from(hyp, off))
            if wav != full:
                wav.unlink(missing_ok=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    outdir = Path(outdir) if outdir else src.parent
    outdir.mkdir(parents=True, exist_ok=True)
    txt_path = outdir / f"{src.stem}.txt"
    srt_path = outdir / f"{src.stem}.srt"
    txt_path.write_text(" ".join(t for t in texts if t) + "\n", encoding="utf-8")
    with srt_path.open("w", encoding="utf-8") as f:
        for i, (a, b, t) in enumerate(segs, 1):
            f.write(f"{i}\n{srt_time(a)} --> {srt_time(b)}\n{t}\n\n")
    return txt_path, srt_path, dur


def main():
    ap = argparse.ArgumentParser(description="Transcribe audio/video with NVIDIA Parakeet TDT 0.6B v2")
    ap.add_argument("inputs", nargs="+", help="audio or video files (any ffmpeg format)")
    ap.add_argument("-o", "--outdir", help="output folder (default: next to each input)")
    ap.add_argument("--local-attn", choices=["auto", "on", "off"], default="auto",
                    help="limited-context attention (auto = on above --full-attn-max-min)")
    ap.add_argument("--full-attn-max-min", type=float, default=4.0,
                    help="longest audio (minutes) done in one full-attention pass (default 4; ~2.6 GB VRAM)")
    ap.add_argument("--auto-chunk-min", type=float, default=5.0,
                    help="piece length (minutes) for longer audio, cut at silences (default 5; ~2.2 GB VRAM)")
    ap.add_argument("--chunk-min", type=float, default=0,
                    help="force N-minute pieces for every file (use 3 if a GPU memory error appears)")
    ap.add_argument("--cpu", action="store_true", help="run on CPU (slow)")
    ap.add_argument("--cuda-graphs", action="store_true",
                    help="use NeMo's CUDA-graph TDT decoder (off by default: under WSL with NeMo 3.0.0 it "
                         "crashes with 'illegal memory access' on the 2nd file of a batch)")
    ap.add_argument("--json", action="store_true", help="print a JSON summary at the end")
    args = ap.parse_args()

    import torch
    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    if device == "cpu" and not args.cpu:
        log("WARNING: CUDA not available, falling back to CPU")
    t0 = time.perf_counter()
    model, dtype = load_model(device, no_cuda_graphs=not args.cuda_graphs)
    log(f"model loaded on {device} ({dtype}) in {time.perf_counter()-t0:.1f}s")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    summary = []
    for inp in args.inputs:
        t1 = time.perf_counter()
        txt, srt, dur = transcribe_file(model, dtype, device, inp, args.outdir, args)
        el = time.perf_counter() - t1
        rtf = dur / el if el > 0 else 0
        log(f"done {Path(inp).name}: {el:.1f}s for {dur:.1f}s audio ({rtf:.0f}x real time)")
        print(str(txt)); print(str(srt))
        summary.append({"input": str(inp), "txt": str(txt), "srt": str(srt),
                        "audio_s": round(dur, 2), "elapsed_s": round(el, 2)})
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 2**20
        log(f"peak PyTorch VRAM: {peak:.0f} MiB")
        for s in summary:
            s["peak_torch_vram_mib"] = round(peak)
    if args.json:
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
