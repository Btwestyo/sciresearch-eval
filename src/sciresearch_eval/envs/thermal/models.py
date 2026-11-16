"""Supplied models (what agents are told) and truth models (what generates the data).

A *supplied model* maps a parameter dict to a Slab. Its free parameters are what an agent can
fit; anything else is given as known. A *truth model* is a JSON-serializable spec that may be
richer than any supplied model (temperature-dependent k, radiation, a hidden contact
resistance, or a finite disk with lateral loss).
"""

from __future__ import annotations

from dataclasses import dataclass

from sciresearch_eval.envs.thermal.axisym import AxisymSettings, Disk, simulate_disk
from sciresearch_eval.envs.thermal.solver import Experiment, Layer, SimulationResult, Slab, SolverSettings, simulate


@dataclass(frozen=True)
class SuppliedModel:
    name: str
    description: str
    parameters: tuple[str, ...]  # all parameters; a task chooses which are free vs known

    def build(self, p: dict[str, float]) -> Slab:
        if self.name in ("slab1", "slab1_kT"):
            layer = Layer(p["L"], p["k"], p["rho_cp"], beta=p.get("beta", 0.0))
            return Slab((layer,), h_rear=p["h"], T_inf=p["T_inf"])
        if self.name == "slab1_rad":
            return Slab((Layer(p["L"], p["k"], p["rho_cp"]),), h_rear=p["h"], T_inf=p["T_inf"],
                        front_emissivity=p["emissivity"])
        if self.name == "slab1_joint":
            half = Layer(p["L"] / 2, p["k"], p["rho_cp"])
            return Slab((half, half), h_rear=p["h"], T_inf=p["T_inf"], contact_resistances=(p["R_c"],))
        if self.name == "slab2":
            layers = (Layer(p["L1"], p["k1"], p["rho_cp1"]), Layer(p["L2"], p["k2"], p["rho_cp2"]))
            return Slab(layers, h_rear=p["h"], T_inf=p["T_inf"], contact_resistances=(p["R_c"],))
        raise ValueError(self.name)


SUPPLIED_MODELS = {
    "slab1": SuppliedModel(
        "slab1",
        "Single homogeneous layer of thickness L with constant conductivity k and volumetric heat "
        "capacity rho_cp. Absorbed heat flux q(t) at the front face; convection with coefficient h to "
        "ambient T_inf at the rear face; no other losses.",
        ("L", "k", "rho_cp", "h", "T_inf"),
    ),
    "slab1_kT": SuppliedModel(
        "slab1_kT",
        "As slab1, but conductivity varies linearly with temperature: k(T) = k * (1 + beta * (T - T_inf)), "
        "with beta known. The solver iterates the nonlinearity each time step.",
        ("L", "k", "rho_cp", "h", "beta", "T_inf"),
    ),
    # Candidate extensions of slab1 (offered together in T6 tasks, so their presence cues no mechanism).
    "slab1_rad": SuppliedModel(
        "slab1_rad",
        "As slab1, plus thermal radiation from the heated front face to surroundings at T_inf with "
        "emissivity epsilon (grey body).",
        ("L", "k", "rho_cp", "h", "emissivity", "T_inf"),
    ),
    "slab1_joint": SuppliedModel(
        "slab1_joint",
        "As slab1, but the sample is two identical halves joined at mid-thickness (x = L/2) with a thermal "
        "contact resistance R_c (m^2 K/W) at the joint.",
        ("L", "k", "rho_cp", "h", "R_c", "T_inf"),
    ),
    "slab2": SuppliedModel(
        "slab2",
        "Two layers (thicknesses L1, L2; properties k1, rho_cp1, k2, rho_cp2) bonded with a thermal "
        "contact resistance R_c. Front-face absorbed flux q(t); rear-face convection h to T_inf.",
        ("L1", "L2", "k1", "rho_cp1", "k2", "rho_cp2", "R_c", "h", "T_inf"),
    ),
}


def run_supplied(model: str, params: dict[str, float], experiment: Experiment, settings: SolverSettings) -> SimulationResult:
    return simulate(SUPPLIED_MODELS[model].build(params), experiment, settings)


# ---------------------------------------------------------------------------
# Truth specs
# ---------------------------------------------------------------------------


def truth_slab_spec(layers: list[dict], h_rear: float, T_inf: float, contact: list[float] | None = None,
                    emissivity: float = 0.0) -> dict:
    return {"kind": "slab", "layers": layers, "h_rear": h_rear, "T_inf": T_inf,
            "contact_resistances": contact or [], "front_emissivity": emissivity}


def truth_disk_spec(radius: float, thickness: float, k: float, rho_cp: float, h_rear: float, h_side: float,
                    T_inf: float) -> dict:
    return {"kind": "disk", "radius": radius, "thickness": thickness, "k": k, "rho_cp": rho_cp,
            "h_rear": h_rear, "h_side": h_side, "T_inf": T_inf}


def run_truth(spec: dict, experiment: Experiment, settings: dict) -> SimulationResult:
    if spec["kind"] == "slab":
        slab = Slab(
            tuple(Layer(**layer) for layer in spec["layers"]),
            h_rear=spec["h_rear"], T_inf=spec["T_inf"],
            contact_resistances=tuple(spec["contact_resistances"]), front_emissivity=spec["front_emissivity"],
        )
        # Ground truth: nonlinear steps by Newton (same equations as Picard, converged to 1e-12 K instead of
        # stopping at picard_tol; 10x fewer iterations). Agent-facing simulations keep Picard.
        return simulate(slab, experiment, SolverSettings(**settings), newton=True)
    if spec["kind"] == "disk":
        disk = Disk(**{k: v for k, v in spec.items() if k != "kind"})
        return simulate_disk(disk, experiment, AxisymSettings(**settings))
    raise ValueError(spec["kind"])
