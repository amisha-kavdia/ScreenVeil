"""NPU Conductor — battery-, latency- and throttle-aware scheduling.

The Conductor answers three questions every frame:

1. **Where** should each model run?  (NPU / GPU / CPU)
   Chosen per workload by *measured* latency x modelled active power, subject
   to the frame's latency budget. Measured at start-up (calibration) and
   refreshed continuously from live timings.
2. **How hard** should the NPU work?  (QNN HTP performance mode)
   burst on AC power, down to low_power_saver when the battery is critical.
3. **How much** work fits in this frame?
   Target FPS, how many screen bands to analyse, how often to run the face
   detector and the background re-verification sweep.

Power modes switch with hysteresis so a battery hovering at a threshold does
not make the pipeline flap. A sustained latency rise on a device (thermal
throttling or contention) is detected against its calibrated baseline and the
Conductor sheds load before frames are dropped.

Energy figures are *estimates* (latency x per-device power coefficient). The
coefficients default to conservative published-order-of-magnitude values and
can be replaced with numbers measured on your own laptop via
`tools/calibrate_power.py` (reads the Windows battery discharge rate).
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger(__name__)

# Modelled active power per device while running inference (watts). Override
# with measured values: ScreenVeilConfig.power_coefficients.
DEFAULT_POWER_W = {"NPU": 1.5, "GPU": 6.0, "CPU": 9.0}


@dataclass(frozen=True)
class Policy:
    target_fps: float
    max_bands: int
    face_every: int          # run face detection on every Nth analysed band batch
    sweep_every: int         # background re-verify one clean band every N frames
    allow: tuple[str, ...]   # devices allowed for heavy models, preference order


POLICIES = {
    "performance": Policy(30, 8, 1, 10, ("NPU", "GPU", "CPU")),
    "balanced": Policy(20, 6, 2, 20, ("NPU", "GPU", "CPU")),
    "saver": Policy(12, 4, 3, 45, ("NPU", "CPU")),
    "critical": Policy(8, 3, 6, 90, ("NPU", "CPU")),
}
MODE_ORDER = ["critical", "saver", "balanced", "performance"]


@dataclass
class PowerSample:
    on_ac: bool | None = None
    percent: float | None = None
    discharge_w: float | None = None      # measured (Windows only), else None


class PowerProbe:
    """Reads battery state. psutil everywhere; discharge watts via WMI on Windows."""

    def __init__(self, poll_s: float = 5.0, read_discharge: bool = True):
        self.poll_s = poll_s
        self._sample = PowerSample()
        self._last = 0.0
        self._discharge: float | None = None
        self._stop = threading.Event()
        import sys
        if read_discharge and sys.platform == "win32":
            threading.Thread(target=self._wmi_loop, daemon=True).start()

    def read(self) -> PowerSample:
        now = time.monotonic()
        if now - self._last >= self.poll_s:
            self._last = now
            try:
                import psutil
                b = psutil.sensors_battery()
            except Exception:
                b = None
            if b is None:                      # desktop / VM: treat as mains power
                self._sample = PowerSample(True, None, None)
            else:
                self._sample = PowerSample(bool(b.power_plugged), float(b.percent), None)
        self._sample.discharge_w = self._discharge
        return self._sample

    def _wmi_loop(self) -> None:  # pragma: no cover - Windows only
        import subprocess
        cmd = ["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus | "
               "Measure-Object -Property DischargeRate -Sum).Sum"]
        while not self._stop.wait(self.poll_s):
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=10,
                                     creationflags=0x08000000).stdout.strip()
                mw = float(out) if out else 0.0
                self._discharge = mw / 1000.0 if mw > 0 else None
            except Exception:
                self._discharge = None


@dataclass
class _Stat:
    ewma: float | None = None
    baseline: float | None = None
    n: int = 0

    def add(self, x: float, alpha: float = 0.2) -> None:
        self.ewma = x if self.ewma is None else (1 - alpha) * self.ewma + alpha * x
        self.n += 1


@dataclass
class TickPlan:
    mode: str
    target_fps: float
    band_budget: int
    run_faces: bool
    do_sweep: bool
    time_budget_ms: float


@dataclass
class Workload:
    key: str
    engine: object               # has devices(), set_device(device, power_mode)
    device: str = "CPU"


class NPUConductor:
    def __init__(self, power: PowerProbe | None = None, forced_mode: str | None = None,
                 power_w: dict[str, float] | None = None, hysteresis: float = 5.0,
                 clock: Callable[[], float] = time.perf_counter):
        self.power = power or PowerProbe()
        self.forced_mode = forced_mode
        self.power_w = {**DEFAULT_POWER_W, **(power_w or {})}
        self.hysteresis = hysteresis
        self.clock = clock
        self.reevaluate_every = 300                      # frames between placement re-checks
        self.workloads: dict[str, Workload] = {}
        self.stats: dict[tuple[str, str], _Stat] = collections.defaultdict(_Stat)
        self.overhead = _Stat()
        self.mode = forced_mode or "balanced"
        self.frame = 0
        self.throttled_ticks = 0
        self.throttle_factor = 1.0
        self.events: collections.deque = collections.deque(maxlen=50)
        self.energy_j = 0.0              # estimated, actual device mix
        self.energy_cpu_only_j = 0.0     # estimated, if everything had run on CPU
        self.busy_s: dict[str, float] = collections.defaultdict(float)
        self._fps_times: collections.deque = collections.deque(maxlen=60)
        self.last_power = PowerSample()

    # ----------------------------------------------------------- registration
    def register(self, key: str, engine) -> None:
        wl = Workload(key, engine, getattr(engine, "device", "CPU"))
        self.workloads[key] = wl

    def calibrate(self, key: str, run_once: Callable[[], None], reps: int = 3) -> dict[str, float]:
        """Measure a workload on every device it supports; sets baselines."""
        wl = self.workloads[key]
        results = {}
        for dev in wl.engine.devices():
            used = wl.engine.set_device(dev, self.mode)
            if used != dev:
                continue
            run_once()                                   # warm-up (graph finalisation)
            times = []
            for _ in range(reps):
                t = self.clock()
                run_once()
                times.append(self.clock() - t)
            lat = sorted(times)[len(times) // 2]
            st = self.stats[(key, dev)]
            st.ewma = st.baseline = lat
            results[dev] = lat
        self._choose_device(wl, force=True)
        self._event(f"calibrated {key}: " + ", ".join(f"{d} {v*1e3:.1f} ms" for d, v in results.items()))
        return results

    # -------------------------------------------------------------- per tick
    def plan(self, pending_age_s: float = 0.0) -> TickPlan:
        self.frame += 1
        self._update_mode()
        if self.frame % self.reevaluate_every == 0:        # live latencies may change the best device
            for wl in self.workloads.values():
                self._choose_device(wl)
        pol = POLICIES[self.mode]
        fps = pol.target_fps * self.throttle_factor
        frame_ms = 1000.0 / fps
        overhead = (self.overhead.ewma or 0.004) * 1000
        budget = max(frame_ms * 0.85 - overhead, 1.0)
        if pending_age_s > 0.4:                          # catch-up: accept a slower frame
            budget *= 1.5
        per_band = self._band_cost_ms()
        n = int(budget // per_band) if per_band > 0 else pol.max_bands
        n = max(1, min(pol.max_bands, n))
        return TickPlan(self.mode, fps, n, self.frame % pol.face_every == 0,
                        self.frame % pol.sweep_every == 0, budget)

    def record(self, key: str, seconds: float) -> None:
        """Report one inference of workload `key` taking `seconds`."""
        wl = self.workloads[key]
        st = self.stats[(key, wl.device)]
        st.add(seconds)
        self.busy_s[wl.device] += seconds
        self.energy_j += seconds * self.power_w.get(wl.device, 5.0)
        cpu = self.stats.get((key, "CPU"))
        dev_base = st.baseline or st.ewma
        cpu_s = seconds * (cpu.baseline / dev_base) if cpu and cpu.baseline and dev_base else seconds
        self.energy_cpu_only_j += cpu_s * self.power_w["CPU"]
        self._check_throttle(key, wl.device, st)

    def record_overhead(self, seconds: float) -> None:
        self.overhead.add(seconds)

    def frame_done(self) -> None:
        self._fps_times.append(self.clock())

    # ------------------------------------------------------------- internals
    def _band_cost_ms(self) -> float:
        total = 0.0
        for key, wl in self.workloads.items():
            st = self.stats.get((key, wl.device))
            if st and st.ewma:
                total += st.ewma * 1000
        return total or 20.0

    def _update_mode(self) -> None:
        if self.forced_mode:
            self.mode = self.forced_mode
            return
        p = self.power.read()
        self.last_power = p
        want = self._mode_for(p)
        if want != self.mode:
            # hysteresis: only step *up* once comfortably past the threshold
            if MODE_ORDER.index(want) > MODE_ORDER.index(self.mode) and p.percent is not None \
                    and not p.on_ac:
                bumped = PowerSample(p.on_ac, p.percent - self.hysteresis)
                if self._mode_for(bumped) == self.mode:
                    return
            self._event(f"power mode {self.mode} -> {want}")
            self.mode = want
            for wl in self.workloads.values():
                self._choose_device(wl, force=True)

    @staticmethod
    def _mode_for(p: PowerSample) -> str:
        if p.on_ac or p.percent is None:
            return "performance"
        if p.percent > 50:
            return "balanced"
        if p.percent > 20:
            return "saver"
        return "critical"

    def _choose_device(self, wl: Workload, force: bool = False) -> None:
        pol = POLICIES[self.mode]
        supported = wl.engine.devices()
        cands = [d for d in pol.allow if d in supported] or list(supported)
        frame_ms = 1000.0 / pol.target_fps
        best, best_cost = None, float("inf")
        fallback, fallback_lat = None, float("inf")
        for d in cands:
            st = self.stats.get((wl.key, d))
            lat = st.ewma if st and st.ewma else None
            if lat is None:
                continue
            if lat < fallback_lat:
                fallback, fallback_lat = d, lat
            if lat * 1000 > frame_ms * 0.6:              # would not fit comfortably
                continue
            cost = lat * self.power_w.get(d, 5.0)        # joules per inference
            if cost < best_cost:
                best, best_cost = d, cost
        choice = best or fallback or cands[0]
        if force or choice != wl.device:
            used = wl.engine.set_device(choice, self.mode)
            if used != wl.device:
                self._event(f"{wl.key}: {wl.device} -> {used} ({self.mode})")
            wl.device = used

    def _check_throttle(self, key: str, dev: str, st: _Stat) -> None:
        if not st.baseline or st.n < 10:
            return
        if st.ewma > 1.6 * st.baseline:
            self.throttled_ticks += 1
            if self.throttled_ticks == 20 and self.throttle_factor > 0.5:
                self.throttle_factor = round(self.throttle_factor * 0.75, 3)
                self._event(f"{key} on {dev} slowed {st.ewma/st.baseline:.1f}x "
                            f"(throttling?) -> fps x{self.throttle_factor}")
                self.throttled_ticks = 0
        else:
            self.throttled_ticks = max(0, self.throttled_ticks - 1)
            if self.throttle_factor < 1.0 and st.ewma < 1.15 * st.baseline:
                self.throttle_factor = min(1.0, round(self.throttle_factor / 0.75, 3))
                self._event(f"{key} recovered -> fps x{self.throttle_factor}")

    def _event(self, msg: str) -> None:
        log.info(msg)
        self.events.append((time.strftime("%H:%M:%S"), msg))

    # ------------------------------------------------------------- telemetry
    def target_fps(self) -> float:
        return POLICIES[self.mode].target_fps * self.throttle_factor

    def fps(self) -> float:
        t = self._fps_times
        return (len(t) - 1) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 0.0

    def snapshot(self) -> dict:
        p = self.last_power
        return {
            "mode": self.mode,
            "on_ac": p.on_ac, "battery_percent": p.percent,
            "measured_discharge_w": p.discharge_w,
            "fps": round(self.fps(), 1),
            "target_fps": round(POLICIES[self.mode].target_fps * self.throttle_factor, 1),
            "workloads": {
                k: {"device": wl.device,
                    "latency_ms": {d: round(s.ewma * 1000, 2)
                                   for (kk, d), s in self.stats.items() if kk == k and s.ewma}}
                for k, wl in self.workloads.items()},
            "busy_s": {d: round(v, 2) for d, v in self.busy_s.items()},
            "energy_est_j": round(self.energy_j, 2),
            "energy_cpu_only_est_j": round(self.energy_cpu_only_j, 2),
            "energy_saved_pct_est": round(100 * (1 - self.energy_j / self.energy_cpu_only_j), 1)
            if self.energy_cpu_only_j > 0 else 0.0,
            "power_coefficients_w": self.power_w,
            "events": list(self.events)[-12:],
        }
