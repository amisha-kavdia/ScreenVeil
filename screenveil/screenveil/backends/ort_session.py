"""ONNX Runtime session factory with Snapdragon (QNN) awareness.

On a Snapdragon X-series Windows PC, install `onnxruntime-qnn`; it exposes the
`QNNExecutionProvider`, which runs models on the Hexagon NPU through the HTP
backend (`QnnHtp.dll`). The NPU Conductor picks the provider and the HTP
performance mode per model at run time; this module turns that choice into a
session.
"""
from __future__ import annotations

import logging
from functools import lru_cache

log = logging.getLogger(__name__)

# Friendly name -> ONNX Runtime provider id
PROVIDER_IDS = {
    "NPU": "QNNExecutionProvider",
    "GPU": "DmlExecutionProvider",
    "CPU": "CPUExecutionProvider",
}

# Conductor power mode -> QNN HTP performance mode
HTP_PERF_MODE = {
    "performance": "burst",
    "balanced": "balanced",
    "saver": "power_saver",
    "critical": "low_power_saver",
}


@lru_cache(maxsize=1)
def available_devices() -> tuple[str, ...]:
    """Friendly device names ONNX Runtime can use on this machine, best first."""
    try:
        import onnxruntime as ort
        have = set(ort.get_available_providers())
    except Exception:  # pragma: no cover - onnxruntime missing
        return ("CPU",)
    return tuple(d for d in ("NPU", "GPU", "CPU") if PROVIDER_IDS[d] in have) or ("CPU",)


def make_session(model_path: str, device: str = "NPU", power_mode: str = "balanced",
                 fp16: bool = True):
    """Create an InferenceSession on `device`, falling back to CPU.

    Returns (session, device_actually_used).
    """
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers: list = []
    if device == "NPU" and "NPU" in available_devices():
        opts = {
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": HTP_PERF_MODE.get(power_mode, "balanced"),
        }
        if fp16:  # lets float32 models run on the HTP in fp16 without a quantisation step
            opts["enable_htp_fp16_precision"] = "1"
        providers.append(("QNNExecutionProvider", opts))
        # Set SCREENVEIL_STRICT_NPU=1 to fail instead of silently running
        # unsupported ops on the CPU (useful when validating an exported model).
        import os
        if os.environ.get("SCREENVEIL_STRICT_NPU") == "1":
            so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    elif device == "GPU" and "GPU" in available_devices():
        providers.append("DmlExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(model_path, sess_options=so, providers=providers)
    used = sess.get_providers()[0]
    used_dev = next((k for k, v in PROVIDER_IDS.items() if v == used), "CPU")
    if used_dev != device:
        log.warning("%s: requested %s, running on %s", model_path, device, used_dev)
    return sess, used_dev
