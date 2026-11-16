"""The optimized solvers must reproduce the frozen v0.2.0 solvers (tests/reference/), so the frozen benchmark's
results stay valid.

- Agent-facing simulations (the supplied and candidate models, Picard iteration): identical, to 1e-9 K, with
  identical step and iteration counts.
- Ground truth (simulated measurements): the 2D disk is identical; nonlinear slabs now use Newton on the same
  equations and agree with the old Picard result to 1e-6 K. The old result carried Picard's own convergence
  error (it stops at a 1e-10 K change per iteration, and that error accumulates over ~1e5 steps). 1e-6 K is
  5e-5 of the smallest sensor noise (0.02 K), below the 1e-5 K rounding of stored task data."""

import json
from pathlib import Path

import numpy as np
import pytest

from sciresearch_eval.envs.thermal import models
from sciresearch_eval.envs.thermal import solver as new
from tests.reference import solver_v020 as ref

ROOT = Path(__file__).resolve().parents[1]
TOL_K = 1e-9
TRUTH_TOL_K = 1e-6


def _to(mod, obj):
    """Rebuild a dataclass instance of `new` as the same-named class of `mod`."""
    cls = getattr(mod, type(obj).__name__)
    vals = {k: _to(mod, v) if hasattr(v, "__dataclass_fields__") else
            tuple(_to(mod, x) for x in v) if isinstance(v, tuple) and v and hasattr(v[0], "__dataclass_fields__") else v
            for k, v in obj.__dict__.items()}
    return cls(**vals)


def compare(slab, exp, settings) -> float:
    a = new.simulate(slab, exp, settings)
    b = ref.simulate(_to(ref, slab), _to(ref, exp), _to(ref, settings))
    for k in ("cells", "steps", "nonlinear_iterations_total", "nonlinear_unconverged_steps", "compute_units"):
        assert a.diagnostics[k] == b.diagnostics[k], k
    assert a.diagnostics["energy_balance_rel_error"] == pytest.approx(b.diagnostics["energy_balance_rel_error"], rel=1e-6, abs=1e-12)
    return max(float(np.max(np.abs(a.readings[s] - b.readings[s]))) for s in a.readings)


def cases(task_file: Path):
    from sciresearch_eval.tasks.generator import experiment_from_dict
    pub = json.loads(task_file.read_text())
    m = pub["supplied_model"]
    p = {**m["known_parameters"], **{q: {"k": 1.3, "rho_cp": 2.1e6, "h": 25.0, "k1": 1.1, "k2": 2.4, "R_c": 2e-3}.get(q, 1.0)
                                     for q in m["free_parameters"]}}
    s = new.SolverSettings(**pub["default_solver_settings"])
    e = experiment_from_dict(pub["initial_data"][0]["experiment"])
    L = p["L"] if "L" in p else p["L1"] + p["L2"]
    inner = tuple(f"x={f * L}" for f in (0.25, 0.5, 0.75))
    yield m["name"], p, e, s
    yield m["name"], p, new.Experiment(new.HeatFlux("pulse", 2 * e.flux.q0, 0.1 * e.duration), e.duration, e.sample_interval, ("front", "rear") + inner), s
    yield m["name"], p, new.Experiment(new.HeatFlux("ramp", e.flux.q0, 0.3 * e.duration), e.duration, e.sample_interval, ("front", "rear"), 0.5 * e.duration), \
        new.SolverSettings(**{**s.__dict__, "theta": 1.0, "cells_per_layer": 30, "dt": s.dt * 3})
    if m["name"] == "slab1":
        for model, extra in (("slab1_kT", {"beta": 4e-3}), ("slab1_rad", {"emissivity": 0.6}), ("slab1_joint", {"R_c": 3e-3})):
            yield model, {**p, **extra}, new.Experiment(new.HeatFlux("step", 3 * e.flux.q0), e.duration, e.sample_interval, ("front", "rear") + inner), s


def sweep(paths) -> float:
    worst = 0.0
    for f in paths:
        for model, p, e, s in cases(f):
            worst = max(worst, compare(models.SUPPLIED_MODELS[model].build(p), e, s))
    return worst


@pytest.mark.parametrize("cells", [1, 2, 3])
@pytest.mark.parametrize("model,extra", [("slab1", {}), ("slab1_kT", {"beta": 4e-3}), ("slab1_rad", {"emissivity": 0.5})])
def test_tiny_meshes_match_reference(cells, model, extra):
    p = {"L": 4e-3, "k": 1.3, "rho_cp": 2.1e6, "h": 25.0, "T_inf": 293.15, **extra}
    e = new.Experiment(new.HeatFlux("step", 2e4), 60.0, 2.0, ("front", "rear", "x=0.002"))
    for theta in (0.5, 1.0):
        assert compare(models.SUPPLIED_MODELS[model].build(p), e, new.SolverSettings(cells_per_layer=cells, dt=0.5, theta=theta)) < TOL_K


def test_matches_reference_on_a_sample_of_tasks():
    paths = sorted((ROOT / "tasks_t10" / "dev" / "public").glob("task_*.json"))[:3] + \
            sorted((ROOT / "tasks" / "dev" / "public").glob("task_*.json"))[::12]
    assert sweep(paths) < TOL_K


@pytest.mark.slow
def test_matches_reference_on_every_task():
    paths = [p for r in ("tasks", "tasks_t10", "tasks_t5", "tasks_t6") for sp in ("dev", "hidden")
             for p in sorted((ROOT / r / sp / "public").glob("task_*.json"))]
    assert sweep(paths) < TOL_K


def _truth_pairs(task_dir: Path, n: int):
    """A few truth runs per task kind: the initial experiment and a strong, long one, at truth settings."""
    from sciresearch_eval.tasks.generator import experiment_from_dict
    for f in sorted((task_dir / "public").glob("task_*.json"))[:n]:
        pub = json.loads(f.read_text())
        priv = json.loads((task_dir / "private" / f.name).read_text())
        e = experiment_from_dict(pub["initial_data"][0]["experiment"])
        yield priv, e
        d = pub["available_designs"]
        yield priv, new.Experiment(new.HeatFlux("step", d["max_flux_W_m2"]), d["max_duration_s"] / 4, d["max_duration_s"] / 150,
                                   ("front", "rear", f"x={0.5 * pub['supplied_model']['known_parameters'].get('L', 0.004)}"))


def _truth_diff(priv, e) -> float:
    from tests.reference import axisym_v020 as ref_disk
    spec, ts = priv["truth_spec"], priv["truth_settings"]
    a = models.run_truth(spec, e, ts)
    if spec["kind"] == "disk":
        b = ref_disk.simulate_disk(ref_disk.Disk(**{k: v for k, v in spec.items() if k != "kind"}), _to(ref, e), ref_disk.AxisymSettings(**ts))
    else:
        slab = new.Slab(tuple(new.Layer(**l) for l in spec["layers"]), h_rear=spec["h_rear"], T_inf=spec["T_inf"],
                        contact_resistances=tuple(spec["contact_resistances"]), front_emissivity=spec["front_emissivity"])
        b = ref.simulate(_to(ref, slab), _to(ref, e), ref.SolverSettings(**ts))
    return max(float(np.max(np.abs(a.readings[s] - b.readings[s]))) for s in a.readings)


@pytest.mark.slow
def test_truth_matches_reference():
    worst = {}
    for root in ("tasks_t10", "tasks_t5"):
        for priv, e in _truth_pairs(ROOT / root / "dev", 16):
            kind = priv["regime"]
            worst[kind] = max(worst.get(kind, 0.0), _truth_diff(priv, e))
    for kind, d in worst.items():
        limit = 1e-9 if ("lateral" in kind or "contact" in kind or "adequate" in kind) else TRUTH_TOL_K
        assert d < limit, (kind, d)
