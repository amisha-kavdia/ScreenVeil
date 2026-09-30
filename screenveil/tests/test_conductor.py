import unittest

from screenveil.conductor import NPUConductor, PowerSample


class FakePower:
    def __init__(self, on_ac=True, percent=None):
        self.s = PowerSample(on_ac, percent)

    def read(self):
        return self.s


class FakeEngine:
    """Pretends to run on NPU/GPU/CPU with configurable latencies (seconds)."""

    def __init__(self, lat):
        self.lat = dict(lat)
        self.device = "CPU"
        self.modes = []

    def devices(self):
        return tuple(self.lat)

    def set_device(self, d, mode):
        self.device = d
        self.modes.append(mode)
        return d


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def calibrated(lat, power=None, **kw):
    clock = Clock()
    c = NPUConductor(power=power or FakePower(), clock=clock, **kw)
    eng = FakeEngine(lat)
    c.register("text", eng)

    def run():
        clock.t += eng.lat[eng.device]

    c.calibrate("text", run)
    return c, eng


class DevicePlacement(unittest.TestCase):
    def test_prefers_npu_when_fast_enough(self):
        c, eng = calibrated({"NPU": 0.012, "GPU": 0.008, "CPU": 0.040})
        self.assertEqual(eng.device, "NPU")        # GPU is faster but costs more energy

    def test_falls_back_to_fastest_when_nothing_fits(self):
        c, eng = calibrated({"NPU": 0.300, "CPU": 0.200})
        self.assertEqual(eng.device, "CPU")

    def test_cpu_only_machine(self):
        c, eng = calibrated({"CPU": 0.02})
        self.assertEqual(eng.device, "CPU")


    def test_placement_reevaluated_from_live_latency(self):
        c, eng = calibrated({"NPU": 0.012, "CPU": 0.040})
        self.assertEqual(eng.device, "NPU")
        for _ in range(40):
            c.record("text", 0.090)                 # NPU became very slow (e.g. contended)
        c.reevaluate_every = 1
        c.plan()
        self.assertEqual(eng.device, "CPU")


class PowerModes(unittest.TestCase):
    def test_modes_follow_battery_with_hysteresis(self):
        p = FakePower(False, 80)
        c, eng = calibrated({"NPU": 0.01, "CPU": 0.03}, power=p)
        self.assertEqual(c.plan().mode, "balanced")
        p.s = PowerSample(False, 45)
        self.assertEqual(c.plan().mode, "saver")
        p.s = PowerSample(False, 52)                # just above threshold: stay (hysteresis)
        self.assertEqual(c.plan().mode, "saver")
        p.s = PowerSample(False, 57)
        self.assertEqual(c.plan().mode, "balanced")
        p.s = PowerSample(False, 10)
        self.assertEqual(c.plan().mode, "critical")
        p.s = PowerSample(True, 10)                 # plugged in: straight to performance
        self.assertEqual(c.plan().mode, "performance")
        self.assertIn("critical", eng.modes)        # NPU perf mode was pushed to the engine

    def test_saver_does_less_work(self):
        p = FakePower(True, None)
        c, _ = calibrated({"NPU": 0.01, "CPU": 0.03}, power=p)
        perf = c.plan()
        p.s = PowerSample(False, 30)
        saver = c.plan()
        self.assertLess(saver.target_fps, perf.target_fps)
        # work per second (bands/frame x frames/s) must drop on battery
        self.assertLess(saver.band_budget * saver.target_fps, perf.band_budget * perf.target_fps)


class ThrottleAndEnergy(unittest.TestCase):
    def test_detects_sustained_slowdown_and_recovers(self):
        c, eng = calibrated({"NPU": 0.010, "CPU": 0.040})
        for _ in range(60):
            c.record("text", 0.025)                 # 2.5x slower than calibrated
        self.assertLess(c.throttle_factor, 1.0)
        for _ in range(60):
            c.record("text", 0.010)
        self.assertEqual(c.throttle_factor, 1.0)

    def test_energy_estimate_vs_cpu_only(self):
        c, eng = calibrated({"NPU": 0.010, "CPU": 0.040})
        for _ in range(10):
            c.record("text", 0.010)
        s = c.snapshot()
        # NPU: 10 x 10 ms x 1.5 W = 0.15 J ; CPU-only: 10 x 40 ms x 9 W = 3.6 J
        self.assertAlmostEqual(s["energy_est_j"], 0.15, places=2)
        self.assertAlmostEqual(s["energy_cpu_only_est_j"], 3.6, places=2)
        self.assertGreater(s["energy_saved_pct_est"], 90)


if __name__ == "__main__":
    unittest.main()
