"""The parameter calculator must be EXACT, not approximate.

Every number in DESIGN.md and RESEARCH.md is generated from ModuleSpec, so it
is checked here against a materialised reference implementation: build the
actual arrays the rollout uses and count them.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nemo.config import ExperimentConfig                      # noqa: E402
from nemo.genome.population import init_population            # noqa: E402
from nemo.modules.spec import ModuleSpec, OrganismSpec        # noqa: E402


def materialised_module_params(d: int, dff: int) -> int:
    """Count the arrays the rollout actually multiplies, independently of spec."""
    return (
        4 * d * d          # Wq, Wk, Wv, Wo
        + d * dff + dff    # W1, b1
        + dff * d + d      # W2, b2
        + 2 * d            # g1, g2 (RMSNorm scales)
        + d + 1            # wg, bg (regulatory head)
    )


@pytest.mark.parametrize("d", [4, 8, 12, 16, 20, 24, 32, 48, 64])
def test_module_param_count_is_exact(d):
    spec = ModuleSpec(d_model=d, d_ff=2 * d)
    assert spec.n_params == materialised_module_params(d, 2 * d)


def test_module_param_breakdown_sums():
    spec = ModuleSpec()
    b = spec.param_breakdown()
    assert b["attention"] + b["ffn"] + b["norm"] + b["gate"] == b["total"]


def test_documented_headline_numbers():
    """The numbers quoted in DESIGN.md must be reproducible from the code."""
    spec = ModuleSpec(d_model=16, d_ff=32, max_in_degree=4)
    assert spec.n_params == 2145
    assert spec.flops_forward() == 7540
    org = OrganismSpec(module=spec, n_genes=4, n_obs_symbols=16, n_actions=8)
    assert org.n_params == 8972


def test_population_arrays_match_spec():
    """The real allocated Population must carry exactly spec.n_params per gene."""
    cfg = ExperimentConfig()
    pop = init_population(cfg, 3, np.zeros(3))
    spec = ModuleSpec(d_model=cfg.module.d_model, d_ff=cfg.module.d_ff,
                      max_in_degree=cfg.module.max_in_degree)
    per_gene = sum(
        getattr(pop, n)[0, 0].size
        for n in ("Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2", "g1", "g2", "wg")
    ) + 1                                        # bg is a scalar per gene
    assert per_gene == spec.n_params


def test_flops_scale_with_in_degree():
    spec = ModuleSpec()
    assert spec.flops_forward(0) < spec.flops_forward(2) < spec.flops_forward(4)
    assert spec.flops_backward() == 2 * spec.flops_forward()


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        ModuleSpec(d_model=15, n_heads=2)
    with pytest.raises(ValueError):
        ModuleSpec(d_model=0)
