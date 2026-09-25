"""Exact parameter and FLOP accounting for the computational module.

The module is deliberately NOT a language-model transformer block.  It is a
*message* block: it consumes a masked set of incoming latent messages, attends
over them from its own recurrent state, and emits one latent message plus one
regulatory (gate) logit.

Rationale for dropping the token vocabulary from the module (see DESIGN.md
"Why modules have no vocabulary"): a per-module embedding table would make
modules non-interchangeable, which destroys the meaning of the duplication,
deletion and rearrangement operators.  Symbol<->latent conversion happens once,
at the organism's sensor and actuator.

Every count here is exact and unit-tested against a materialised reference
implementation (tests/test_param_count.py), so no number in the design docs is
an estimate.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSpec:
    """Shape of one computational module (one gene's product)."""

    d_model: int = 16
    d_ff: int = 32
    n_heads: int = 1
    max_in_degree: int = 4          # K: incoming messages attended over
    attn_bias: bool = False
    ffn_bias: bool = True
    gate: bool = True               # emits a regulatory logit

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} not divisible by n_heads={self.n_heads}")
        for name in ("d_model", "d_ff", "n_heads", "max_in_degree"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1, got {getattr(self, name)}")

    # ---- parameters -----------------------------------------------------
    @property
    def p_attention(self) -> int:
        """W_q, W_k, W_v, W_o, each d_model x d_model (+ optional biases)."""
        d = self.d_model
        n = 4 * d * d
        if self.attn_bias:
            n += 4 * d
        return n

    @property
    def p_ffn(self) -> int:
        """d_model -> d_ff -> d_model."""
        n = 2 * self.d_model * self.d_ff
        if self.ffn_bias:
            n += self.d_ff + self.d_model
        return n

    @property
    def p_norm(self) -> int:
        """Two RMSNorm scale vectors (pre-attn and pre-ffn)."""
        return 2 * self.d_model

    @property
    def p_gate(self) -> int:
        """Regulatory head: scalar logit from the post-attention state."""
        return (self.d_model + 1) if self.gate else 0

    @property
    def n_params(self) -> int:
        return self.p_attention + self.p_ffn + self.p_norm + self.p_gate

    def param_breakdown(self) -> dict[str, int]:
        return {
            "attention": self.p_attention,
            "ffn": self.p_ffn,
            "norm": self.p_norm,
            "gate": self.p_gate,
            "total": self.n_params,
        }

    # ---- FLOPs ----------------------------------------------------------
    def flops_forward(self, in_degree: int | None = None) -> int:
        """Multiply-accumulate FLOPs (2 per MAC) for one module-step.

        in_degree defaults to max_in_degree; a module whose realised in-degree
        is lower costs proportionally less, which is what the metabolic
        accounting charges for.
        """
        k = self.max_in_degree if in_degree is None else in_degree
        if k < 0:
            raise ValueError("in_degree must be >= 0")
        d, f = self.d_model, self.d_ff
        q = 2 * d * d                    # query from own state
        kv = 2 * (2 * k * d * d)         # keys and values for k messages
        scores = 2 * k * d               # q . K
        softmax = 5 * k                  # exp/sum/div, generous
        weighted = 2 * k * d             # a @ V
        out = 2 * d * d                  # W_o
        ffn = 2 * (2 * d * f)            # two GEMMs
        norms = 4 * d                    # two RMSNorms
        gate = 2 * d                     # dot + bias
        return q + kv + scores + softmax + weighted + out + ffn + norms + gate

    def flops_backward(self, in_degree: int | None = None) -> int:
        """Backward is ~2x forward for dense layers (grad wrt input + weights)."""
        return 2 * self.flops_forward(in_degree)

    def bytes_bf16(self) -> int:
        return 2 * self.n_params


@dataclass(frozen=True)
class OrganismSpec:
    """Organism-level (non-module) parameters: the sensory/motor interface."""

    module: ModuleSpec = ModuleSpec()
    n_obs_symbols: int = 16     # environment observation alphabet
    n_actions: int = 8
    n_genes: int = 4            # ancestral genome length
    obs_slots: int = 2          # observation channels per timestep

    @property
    def p_sensor(self) -> int:
        """Embedding table mapping observation symbols to latent messages."""
        return self.n_obs_symbols * self.module.d_model

    @property
    def p_actuator(self) -> int:
        """Linear readout to action logits."""
        return self.module.d_model * self.n_actions + self.n_actions

    @property
    def p_interface(self) -> int:
        return self.p_sensor + self.p_actuator

    @property
    def p_modules(self) -> int:
        return self.n_genes * self.module.n_params

    @property
    def n_params(self) -> int:
        return self.p_modules + self.p_interface

    def regulatory_genome_size(self) -> int:
        """Scalars in the non-neural genome (wiring + regulation + meta).

        Per gene: module slot ref, K input source refs, execution round,
        gate bias, gate slope, baseline activation propensity.
        Plus 8 organism-level heritable mutation-rate parameters.
        """
        per_gene = 1 + self.module.max_in_degree + 1 + 3
        return self.n_genes * per_gene + 8

    def breakdown(self) -> dict[str, int]:
        return {
            "modules": self.p_modules,
            "sensor": self.p_sensor,
            "actuator": self.p_actuator,
            "neural_total": self.n_params,
            "regulatory_scalars": self.regulatory_genome_size(),
        }
