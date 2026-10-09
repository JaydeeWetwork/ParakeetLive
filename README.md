# Parakeet Live

Local, private, GPU-accelerated live dictation for Windows. A small always-on-top widget records
your microphone; each time you pause, that stretch of speech is transcribed by
**NVIDIA Parakeet TDT 0.6B v2** (through NVIDIA NeMo) running inside **WSL2**, and the text lands in
the widget, already copied to the clipboard, ready to paste into any app.

There is also a batch transcriber for audio/video files (punctuated `.txt` plus timestamped `.srt`).

Everything runs on your own PC. There is no cloud service, no account and no telemetry.

Tested on a few-years-old mid-range gaming laptop: Windows 11, NVIDIA RTX 3050 Ti Laptop GPU (4 GB VRAM).

## Features

- **Live dictation widget:** click the round button or press the global hotkey **Ctrl+Alt+Space**
  (from any app) to start or stop listening. After each pause (0.5 s by default, 0.3-1.5 s
  selectable) the speech is transcribed and appended. English only (the model is English).
- **Auto-copy:** the whole box is copied to the clipboard after every new piece of text and shortly
  after you stop typing in it. The box is fully editable.
- **Fresh message after pasting:** once the text was copied and you switched to another window (or
  pressed Ctrl+V), the next recording starts a new message. Ctrl+Z in the box or
  "Restore last message" brings the old text back. Both behaviours can be turned off.
- **Shows without stealing focus:** if the widget is hidden when the hotkey starts a recording, it
  appears on top without taking keyboard focus.
- **Tray icon** with the model state: not loaded, in RAM standby, ready on the GPU, recording.
- **GPU-friendly:** the model stays on the GPU (about 1.4 GB VRAM) for instant dictation, and is
  automatically **parked in RAM** while a game or another GPU-heavy app needs the card, then moved
  back (about 60 s after that app is gone). Optional RAM standby, idle unload and pre-load at login.
- **Batch transcription** of long files via PowerShell; it asks a running widget to move its model off
  the GPU for the duration of the job.
- **Robust:** a failed transcription is retried and its audio kept for up to 1 hour (also across a
  restart), a stalled microphone is detected, and a
  crashed engine restarts by itself.

## How it works

```
widget\parakeet_live.pyw  (Windows, Python 3.14, tkinter + ctypes, sounddevice for the mic)
   |  utterance audio over HTTP, 127.0.0.1:51761, per-run random secret
   v
server\live_server.py     (inside WSL Ubuntu-24.04, /opt/parakeet/venv, PyTorch + NeMo on the GPU)
```

The widget starts the server inside WSL on demand. The server listens only on 127.0.0.1, refuses
requests from web pages, caps request size and requires a secret the widget generates fresh on every
start (passed via the environment, never on disk).

## Requirements

- **Windows 11** (64-bit) with **WSL2**.
- A WSL distro named exactly **`Ubuntu-24.04`** (the name is set in `widget\plive_core.py` and
  `batch\Transcribe-Parakeet.ps1`). The setup and the server run as root inside that distro and
  install everything under `/opt/parakeet`.
- An **NVIDIA GPU with about 4 GB VRAM** or more (live model about 1.4 GB; a batch job uses up to
  about 2.7 GB) and a recent **Windows NVIDIA driver** (WSL gets CUDA from the Windows driver; no
  Linux driver or CUDA toolkit is installed in WSL).
- **Windows Python 3.14** for the widget (the pins in `widget\requirements.txt` were tested with 3.14).
- Inside WSL: Python 3.12 (Ubuntu 24.04's default) and ffmpeg.
- Disk: plan for roughly 10 GB inside WSL (PyTorch with CUDA, NeMo, the 2.3 GB model, caches).

## Install

All commands run in PowerShell. Steps 4-6 run from the repo folder.

1. **WSL and Ubuntu 24.04**

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

2. **System packages**

   ```powershell
   wsl -d Ubuntu-24.04 -u root -e bash -c "apt update && apt install -y python3.12-venv python3-dev ffmpeg"
   ```

3. **Python venv and PyTorch (CUDA 12.6 build)**

   ```powershell
   wsl -d Ubuntu-24.04 -u root -e bash -c "mkdir -p /opt/parakeet && python3 -m venv /opt/parakeet/venv && /opt/parakeet/venv/bin/pip install -U pip && /opt/parakeet/venv/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126"
   ```

4. **NeMo and the cache layout** (keeps all downloads under `/opt/parakeet`, not `/root/.cache`)

   ```powershell
   wsl -d Ubuntu-24.04 -u root -e bash tools/setup-nemo.sh
   ```

5. **Download the model** (`nvidia/parakeet-tdt-0.6b-v2` from Hugging Face, about 2.3 GB)

   ```powershell
   wsl -d Ubuntu-24.04 -u root -e bash tools/fetch-model.sh
   ```

6. **Deploy the server and batch scripts** into `/opt/parakeet/scripts` (run again after you change
   `server\live_server.py`, `server\plive_fastload.py`, `batch\transcribe.py` or `batch\env.sh`)

   ```powershell
   wsl -d Ubuntu-24.04 -u root -e bash tools/deploy-to-wsl.sh
   ```

   Optional, for faster model loading: build the pre-extracted checkpoint once (re-run it after
   replacing the model). Without it the server loads the `.nemo` file the normal, slower way.

   ```powershell
   wsl -d Ubuntu-24.04 -u root -e /opt/parakeet/venv/bin/python server/build-fast-checkpoint.py
   ```

   `server/verify-fast-load.py <folder with 16 kHz mono .wav clips>` checks that the fast loader gives
   bit-identical weights and identical transcripts.

7. **Widget venv (Windows)**

   ```powershell
   py -3.14 -m venv widget\.venv
   widget\.venv\Scripts\python.exe -m pip install -r widget\requirements.txt
   ```

8. **Run it**

   ```powershell
   widget\.venv\Scripts\pythonw.exe widget\parakeet_live.pyw
   ```

   Make a shortcut to that command for the Start Menu. To start it at login as a tray icon only, put a
   shortcut with `--tray` added in your Startup folder (`shell:startup`); it loads the model onto the
   GPU right away (RAM first only while a game or batch job holds the GPU), so dictation is ready soon after login.

`tools/versions.sh` prints the installed versions inside WSL.

## Usage

- **Round button** (left): click to start listening, click again to stop. **Ctrl+Alt+Space** does the
  same from any app.
- Speak, pause, and the text appears. Paste it wherever you want (it is already on the clipboard).
- **Top bar:** copy all, clear, gear (settings), minimize and X. Minimize, X and **Esc** hide the
  widget to the tray (the model stays loaded). **Quit** is in the tray menu.
- **Tray icon:** left-click shows or hides the widget. Right-click for recording, model loading and
  unloading, pause auto-park, save training data, settings and quit. New tray icons often land in the
  ^ overflow; drag it onto the taskbar to keep it visible.
- Drag the widget by its frame or the round button; resize it with the grip in the bottom-right
  corner. Position, size, microphone and settings are remembered.
- **Command line** (talks to the running copy):
  `pythonw parakeet_live.pyw --cmd show|hide|toggle|load|unload|record|prewarm|standby|pausepark|quit`

### Batch transcription

```powershell
powershell -ExecutionPolicy Bypass -File batch\Transcribe-WithGpuHold.ps1 "C:\path\to\recording.mp3"
```

Writes `recording.txt` and `recording.srt` next to the input. Any format ffmpeg reads works.
`Transcribe-WithGpuHold.ps1` asks a running widget to park its model in RAM for the job and gives it
back afterwards; `batch\Transcribe-Parakeet.ps1` is the plain transcriber underneath.

| Option | Meaning |
|---|---|
| `-OutDir C:\Transcripts` | write outputs to another folder |
| several paths | transcribe a batch in one run (the model loads once) |
| `-ChunkMin 3` | force 3-minute pieces (only if you get a GPU out-of-memory error) |
| `-LocalAttn on/off/auto` | limited-context attention (auto = on for audio over 4 min) |
| `-Cpu` | run on the CPU (slow) |
| `-CudaGraphs` | re-enable NeMo's CUDA-graph decoder (off on purpose) |

Each run spends about 20-40 s starting WSL and loading the model, so pass several files at once.

## Configuration

Most settings are in the gear menu (or tray > Settings). They are saved in `widget\config.json`
(created on first run, not tracked by git). Quit the widget before editing it by hand. Useful keys:

| Key | Default | Meaning |
|---|---|---|
| `hotkey` | `ctrl+alt+space` | global start/stop hotkey |
| `silence_s` | `0.5` | pause length that ends an utterance |
| `autocopy` | `true` | copy the whole box after each change |
| `save_training` | `true` | keep audio + text of each utterance locally (see Privacy) |
| `gpu_always` / `auto_park` | `true` / `true` | keep the model on the GPU; park it while another app needs the GPU |
| `park_games` | Minecraft processes | park at once while one of these runs |
| `park_ignore` | `[]` | process names that never trigger auto-park |
| `port` | `51761` | local port of the WSL server |
| `data_dir` | see below | data folder |

**Data folder:** the `PARAKEET_LIVE_DATA` environment variable, else `data_dir` in `config.json`,
else `%LOCALAPPDATA%\ParakeetLive`. It holds `logs\` (widget.log, server.log, state.json),
`state\` (audio waiting for a retry) and `training-data\`.

## Privacy

- Audio and text stay on your PC. The server listens only on 127.0.0.1, and Parakeet Live itself
  sends nothing anywhere (no telemetry). The network is only used during install (apt, pip and the
  Hugging Face model download).
- `widget.log` and `server.log` record timings and the *length* of each transcript, never the text.
- **Training data is saved by default.** With "Save training data" on, every utterance's audio
  (16 kHz WAV, about 115 MB per hour of dictation) and its text go into `training-data\` in the data
  folder, in a NeMo-compatible `manifest.jsonl`, for your own fine-tuning later. The app never
  deletes these files. Turn it off in the tray or gear menu if you don't want it.
- Every launch starts with an empty box; the box text is never written to disk. Audio of a failed
  transcription is kept in `state\pending\` for up to 1 hour so it can be retried.
- Auto-copy puts every version of the box on the clipboard, so Windows clipboard history (Win+V)
  may keep them. Turn off Auto-copy and use Copy all if that bothers you.
- To detect pasting, the widget checks just the Ctrl and V key states, and only while copied text is
  waiting and another window is in front. It installs no keyboard hook and records no keys.
  Turn it off with the gear menu option "Ctrl+V counts as pasting".
  **Ctrl+V while recording stops the recording** (same as pressing stop): words still being
  transcribed stay in the box, and the pasted message is cleared from the box (Ctrl+Z or "Restore last
  message" brings it back). This uses the same Ctrl+V check, so no keys are read at any other time.
  While recording, this also works after you switched to the other window (for example copy in the
  widget, click into a chat box, Ctrl+V). A paste with the mouse (right-click > Paste) is not seen.

## Troubleshooting

- **No text / the button shows an error:** Windows Settings > Privacy & security > Microphone:
  "Microphone access" and "Let desktop apps access your microphone" must be on. Pick the right input
  under gear > Microphone.
- **"Speech engine stopped/not answering":** with "Keep model on GPU" it restarts by itself (at most
  twice in 10 minutes); otherwise tray > Load model. Details are in `logs\server.log` in the data folder.
- **First load is slow:** expect about 25 s on a cold start (WSL boot + model load); the RAM pre-load
  at login and the fast checkpoint (install step 6) make later loads much quicker.
- **The model went to RAM while gaming:** that is auto-park; the tray tooltip names the app. "Pause
  auto-park" forces it back. Add harmless apps to `park_ignore`.
- **Uses too much RAM:** turn off "Keep model on GPU", "Pre-load at login" and "Keep model in RAM
  when idle", or tray > Unload model completely.
- **"Not copied" (red):** another app held the clipboard; click Copy all or keep dictating.
- **Settings back to defaults:** an unreadable `config.json` is set aside as
  `config.json.corrupt-<time>` and the defaults are used.
- **Widget jumped to the bottom centre of the main screen:** its monitor was disconnected; it moves
  back when that monitor returns.
- A red `OneLogger` / `NativeCommandError` line at the start of a batch run is NeMo writing a notice
  to stderr; it is harmless.
- **Remove everything:** `wsl --unregister Ubuntu-24.04` deletes the whole distro (only if you use it
  for nothing else), then delete the repo folder and the data folder.

## Tests

`tests\` holds the developer tests. Some drive a real widget in an invisible, click-through sandbox
and need the full WSL setup plus test clips in `test-samples\` in the data folder; others are plain
unit tests (for example `tests\test_core_units.py`, `tests\test_park_policy.py`).

## Tested versions

| Component | Version |
|---|---|
| Windows | 11, NVIDIA driver 617.42 |
| WSL | 2.7.14 |
| Distro | Ubuntu 24.04.5 LTS (`Ubuntu-24.04`) |
| Python (WSL venv) | 3.12.3 |
| PyTorch | 2.14.1+cu126, torchaudio 2.11.0 |
| NeMo | nemo_toolkit 3.0.0 (`[asr]`) |
| Model | nvidia/parakeet-tdt-0.6b-v2 |
| ffmpeg | 6.1.1 (Ubuntu package) |
| Widget | Python 3.14; sounddevice 0.5.6, cffi 2.1.1, pycparser 3.0, numpy 2.5.3, pillow 12.3.0 |

## License and credits

The code in this repository is released under the [MIT License](LICENSE).
Copyright (c) 2026 Jaydee Wetwork.

Third-party components (none of them are included in this repository; they are downloaded during
setup from their official sources):

| Component | License | Notes |
|---|---|---|
| [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2) | CC-BY-4.0 | speech recognition model by NVIDIA; downloaded from Hugging Face at setup, not redistributed |
| [NVIDIA NeMo](https://github.com/NVIDIA/NeMo) | Apache-2.0 | toolkit that loads and runs the model (in WSL) |
| [PyTorch](https://pytorch.org/) | BSD-3-Clause | in WSL |
| [sounddevice](https://github.com/spatialaudio/python-sounddevice) 0.5.6 | MIT | microphone input; its Windows wheel bundles PortAudio (MIT-style license) |
| [cffi](https://github.com/python-cffi/cffi) 2.1.1 | MIT | dependency of sounddevice |
| [pycparser](https://github.com/eliben/pycparser) 3.0 | BSD-3-Clause | dependency of cffi |
| [NumPy](https://numpy.org/) 2.5.3 | BSD-3-Clause | audio buffers |
| [Pillow](https://python-pillow.org/) 12.3.0 | MIT-CMU (HPND) | icons and drawing |
| [FFmpeg](https://ffmpeg.org/) | LGPL/GPL (Ubuntu package) | audio conversion in WSL |

Parakeet TDT 0.6B v2 is provided by NVIDIA under the Creative Commons Attribution 4.0 license; see
its model card for details. This project is not affiliated with or endorsed by NVIDIA.