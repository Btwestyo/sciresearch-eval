"""Verification of the 2D axisymmetric disk solver used for lateral-loss ground truth."""

import numpy as np
import pytest

from sciresearch_eval.envs.thermal.axisym import AxisymSettings, Disk, simulate_disk
from sciresearch_eval.envs.thermal.solver import Experiment, HeatFlux, Layer, Slab, SolverSettings, simulate

EXP = Experiment(HeatFlux("step", 5.0e3), 120.0, 2.0, ("front", "x=0.002", "rear"))


def test_no_side_loss_reduces_to_1d():
    disk = Disk(radius=0.0125, thickness=0.004, k=1.0, rho_cp=2.0e6, h_rear=20.0, h_side=0.0)
    two_d = simulate_disk(disk, EXP, AxisymSettings(nr=8, nz=60, dt=0.1))
    one_d = simulate(Slab((Layer(0.004, 1.0, 2.0e6),), h_rear=20.0), EXP, SolverSettings(cells_per_layer=60, dt=0.1))
    for name in EXP.sensors:
        assert np.max(np.abs(two_d.readings[name] - one_d.readings[name])) < 1e-6


def test_energy_balance_with_side_loss():
    disk = Disk(radius=0.0125, thickness=0.004, k=1.0, rho_cp=2.0e6, h_rear=20.0, h_side=50.0)
    res = simulate_disk(disk, EXP, AxisymSettings(nr=30, nz=40, dt=0.1))
    assert res.diagnostics["energy_balance_rel_error"] < 1e-9


def test_side_loss_lowers_axis_temperatures_and_converges():
    lossy = Disk(radius=0.005, thickness=0.004, k=1.0, rho_cp=2.0e6, h_rear=20.0, h_side=50.0)
    sealed = Disk(radius=0.005, thickness=0.004, k=1.0, rho_cp=2.0e6, h_rear=20.0, h_side=0.0)
    s = AxisymSettings(nr=40, nz=40, dt=0.1)
    a, b = simulate_disk(lossy, EXP, s).readings["rear"], simulate_disk(sealed, EXP, s).readings["rear"]
    assert a[-1] < b[-1] - 0.1  # measurable deficit (K)
    fine = simulate_disk(lossy, EXP, AxisymSettings(nr=80, nz=80, dt=0.05)).readings["rear"]
    assert np.max(np.abs(a - fine)) < 0.02 * np.max(fine - 293.15)  # resolution adequate at 40x40
