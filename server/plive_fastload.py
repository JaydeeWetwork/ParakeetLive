"""Fast Parakeet loading for the live server (deployed to /opt/parakeet/scripts/plive_fastload.py).

The stock path (ASRModel.restore_from(.nemo)) untars the 2.4 GB .nemo into a temp dir, torch.load()s the fp32
pickle and randomly initialises every layer before overwriting it. Here:
  * the .nemo is pre-extracted once into FAST_DIR (config + tokenizer files) by build-fast-checkpoint.py,
  * the weights are a safetensors file with the encoder already in bfloat16 (1.2 GB instead of 2.4 GB,
    memory-mapped read; same values the GPU path uses anyway: fp32 -> bf16 cast is identical),
  * random weight init is skipped during construction (every parameter and persistent buffer is then
    overwritten by load_state_dict(strict=True); verified bit-identical by verify-fast-load.py).
NeMo's own SaveRestoreConnector does the rest (model_extracted_dir is a supported NeMo feature)."""
import contextlib
import json
import os

FAST_DIR = "/opt/parakeet/models/parakeet-tdt-0.6b-v2-fast"
NEMO_FILE = "/opt/parakeet/models/parakeet-tdt-0.6b-v2.nemo"
WEIGHTS = "model_weights.safetensors"
_INIT_FUNCS = ("uniform_", "normal_", "trunc_normal_", "constant_", "ones_", "zeros_", "eye_", "dirac_",
               "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", "kaiming_normal_", "orthogonal_", "sparse_")


def available():
    try:
        with open(os.path.join(FAST_DIR, "manifest.json")) as f:
            m = json.load(f)
        return m.get("complete") is True and os.path.exists(os.path.join(FAST_DIR, WEIGHTS))
    except Exception:
        return False


@contextlib.contextmanager
def skip_weight_init():
    import torch
    saved = {n: getattr(torch.nn.init, n) for n in _INIT_FUNCS if hasattr(torch.nn.init, n)}
    noop = lambda tensor, *a, **k: tensor  # noqa: E731
    try:
        for n in saved:
            setattr(torch.nn.init, n, noop)
        yield
    finally:
        for n, f in saved.items():
            setattr(torch.nn.init, n, f)


def load_cpu(skip_init=True):
    """Model on CPU, eval mode, encoder in bfloat16."""
    import torch
    import nemo.collections.asr as nemo_asr
    from nemo.core.connectors.save_restore_connector import SaveRestoreConnector
    import safetensors.torch as st

    class FastConnector(SaveRestoreConnector):
        @staticmethod
        def _load_state_dict_from_disk(model_weights, map_location="cpu"):
            return st.load_file(model_weights, device="cpu")

    con = FastConnector()
    con.model_extracted_dir = FAST_DIR
    con.model_weights_ckpt = WEIGHTS
    ctx = skip_weight_init() if skip_init else contextlib.nullcontext()
    with ctx:
        model = nemo_asr.models.ASRModel.restore_from(NEMO_FILE, map_location="cpu", strict=True,
                                                      save_restore_connector=con)
    model.eval()
    model.encoder.to(torch.bfloat16)
    return model
