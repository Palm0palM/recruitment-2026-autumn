#!/usr/bin/env python3
"""Guard against accidentally timing Python output readback again."""
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import final_bench


class TimingContract(unittest.TestCase):
    def test_readback_happens_after_clock_stop(self):
        events = []

        def clock():
            events.append("clock")
            return float(len(events))

        def process(*args, **kwargs):
            events.append("process")
            return SimpleNamespace(returncode=0, stderr=b"")

        def readback(*args, **kwargs):
            events.append("readback")
            return np.array([1], dtype=np.float32)

        with patch.object(final_bench.time, "perf_counter", clock), \
             patch.object(final_bench.subprocess, "run", process), \
             patch.object(final_bench.np, "fromfile", readback):
            elapsed, _, _ = final_bench.invoke("program", Path("input"),
                                              Path("output"), {}, "cpu")
        self.assertEqual(events, ["clock", "process", "clock", "readback"])
        self.assertEqual(elapsed, 2.0)


if __name__ == "__main__":
    unittest.main()
