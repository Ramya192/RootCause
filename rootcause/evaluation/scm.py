"""A small structural causal model (SCM) that can be sampled AND intervened on.

This is what makes a true ATE computable: the data-generating process is
declared once, as code, so the evaluation harness can draw the observational
data the pipeline sees and also simulate do() on the very same mechanisms.
(`ground_truth.json` only records direct coefficients, which are not total
effects: a treatment's effect flows through mediators and, here, a sigmoid.)

Each node is a function of its parents' already-drawn values plus its own
noise. Nodes are evaluated in the order given, which must be topological.
`sample` draws every node's noise from one RNG in that fixed order, even for
a node that is being intervened on -- so a run with an intervention and a run
without one that share a seed use the same exogenous noise (common random
numbers). The difference between them is then the causal effect with almost no
Monte Carlo variance, instead of the difference of two independent noisy means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import numpy as np
import pandas as pd

# (values of all earlier nodes, rng, n) -> this node's values
Mechanism = Callable[[Mapping[str, np.ndarray], np.random.Generator, int], np.ndarray]


@dataclass(frozen=True)
class Node:
    name: str
    parents: tuple[str, ...]
    mechanism: Mechanism
    # A latent node is simulated but withheld from the sampled DataFrame (a
    # hidden confounder): the pipeline never sees it, and it is not part of
    # the observed ground-truth graph.
    latent: bool = False


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def linear_gaussian(intercept: float = 0.0, noise_sd: float = 1.0, **coefs: float) -> tuple[tuple[str, ...], Mechanism]:
    """`intercept + sum(coef * parent) + N(0, noise_sd)`. Returns (parents, mechanism)."""

    def mechanism(values, rng, n):
        signal = intercept + sum(c * values[p] for p, c in coefs.items())
        return signal + rng.normal(0.0, noise_sd, n)

    return tuple(coefs), mechanism


def additive_noise(
    parents: tuple[str, ...], fn: Callable[[Mapping[str, np.ndarray]], np.ndarray], noise_sd: float = 1.0
) -> tuple[tuple[str, ...], Mechanism]:
    """`fn(values) + N(0, noise_sd)` for an arbitrary (e.g. nonlinear) `fn`.
    `parents` must list every earlier node `fn` reads. Returns (parents, mechanism)."""

    def mechanism(values, rng, n):
        return fn(values) + rng.normal(0.0, noise_sd, n)

    return parents, mechanism


def logistic_binary(intercept: float = 0.0, logit_noise_sd: float = 0.0, **coefs: float) -> tuple[tuple[str, ...], Mechanism]:
    """Bernoulli(sigmoid(intercept + sum(coef * parent) + N(0, logit_noise_sd))).

    Draws the logit noise (only if `logit_noise_sd` > 0) and then a uniform, in
    that order, so the RNG stream is reproducible.
    """

    def mechanism(values, rng, n):
        logit = intercept + sum(c * values[p] for p, c in coefs.items())
        if logit_noise_sd:
            logit = logit + rng.normal(0.0, logit_noise_sd, n)
        return (rng.uniform(0.0, 1.0, n) < sigmoid(logit)).astype(int)

    return tuple(coefs), mechanism


def logistic_of(
    parents: tuple[str, ...], fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
) -> tuple[tuple[str, ...], Mechanism]:
    """Bernoulli(sigmoid(fn(values))) for an arbitrary logit `fn` (e.g. one that
    uses log1p of a parent, which `logistic_binary`'s linear form cannot)."""

    def mechanism(values, rng, n):
        return (rng.uniform(0.0, 1.0, n) < sigmoid(fn(values))).astype(int)

    return parents, mechanism


def bernoulli_of(
    parents: tuple[str, ...], fn: Callable[[Mapping[str, np.ndarray]], np.ndarray]
) -> tuple[tuple[str, ...], Mechanism]:
    """Bernoulli(p) where p = fn(values) (a probability, not a logit). A constant
    `p` needs no parents: `bernoulli_of((), lambda v: 0.4)` -- `fn` may return a scalar."""

    def mechanism(values, rng, n):
        return (rng.uniform(0.0, 1.0, n) < fn(values)).astype(int)

    return parents, mechanism


@dataclass(frozen=True)
class SCM:
    nodes: tuple[Node, ...]
    # False when the covariates are real data (a semi-synthetic SCM): their mutual structure is
    # not known, so `edges()` is incomplete and Stage 3 cannot be scored against it. Effects
    # (do() on the nodes that ARE modelled) remain valid.
    graph_known: bool = True
    _index: dict[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index: dict[str, int] = {}
        for i, node in enumerate(self.nodes):
            if node.name in index:
                raise ValueError(f"duplicate node {node.name!r}")
            for parent in node.parents:
                if parent not in index:
                    raise ValueError(
                        f"node {node.name!r} lists parent {parent!r} that is not defined "
                        "earlier -- nodes must be in topological order"
                    )
            index[node.name] = i
        object.__setattr__(self, "_index", index)

    @property
    def observed(self) -> list[str]:
        return [n.name for n in self.nodes if not n.latent]

    def edges(self) -> list[tuple[str, str]]:
        """Directed parent->child edges among OBSERVED nodes. A path through a
        latent node is not an edge, and a latent common cause is not represented."""
        seen = set(self.observed)
        return [
            (parent, node.name)
            for node in self.nodes
            if node.name in seen
            for parent in node.parents
            if parent in seen
        ]

    def sample(
        self,
        n: int,
        seed: int,
        shift: Optional[Mapping[str, float]] = None,
        set_to: Optional[Mapping[str, float]] = None,
    ) -> pd.DataFrame:
        """Draw `n` rows. `shift` adds a constant to a node's natural value;
        `set_to` replaces it outright (hard do()). Either way children see the
        intervened value. Latent nodes are dropped from the result."""
        shift, set_to = dict(shift or {}), dict(set_to or {})
        for name in [*shift, *set_to]:
            if name not in self._index:
                raise KeyError(f"cannot intervene on unknown node {name!r}")
        overlap = shift.keys() & set_to.keys()
        if overlap:
            raise ValueError(f"node(s) {sorted(overlap)} given both shift and set_to")

        rng = np.random.default_rng(seed)
        values: dict[str, np.ndarray] = {}
        for node in self.nodes:
            drawn = node.mechanism(values, rng, n)  # always drawn, for common random numbers
            if node.name in set_to:
                drawn = np.full(n, set_to[node.name])
            elif node.name in shift:
                drawn = drawn + shift[node.name]
            values[node.name] = drawn
        return pd.DataFrame({name: values[name] for name in self.observed})

    def true_effect(
        self,
        treatment: str,
        outcome: str,
        shift: float = 1.0,
        n: int = 2_000_000,
        seed: int = 0,
    ) -> float:
        """E[outcome | do(treatment := natural value + shift)] - E[outcome].

        For a binary outcome this is a change in probability. It is the
        population-average effect of moving everyone's `treatment` by `shift`,
        which is exactly the quantity Stage 6 forms as `ate * expected_shift`.
        Monte Carlo over `n` paired draws; standard error is well under 1e-3.
        """
        if outcome not in self.observed:
            raise KeyError(f"outcome {outcome!r} is not an observed node")
        base = self.sample(n, seed)[outcome].to_numpy(dtype=float)
        moved = self.sample(n, seed, shift={treatment: shift})[outcome].to_numpy(dtype=float)
        return float(moved.mean() - base.mean())

    def is_binary(self, name: str, n: int = 2000, seed: int = 0) -> bool:
        """Whether an observed node only takes the values 0 and 1 (checked on a draw)."""
        return set(np.unique(self.sample(n, seed)[name])) <= {0, 1}

    def true_effect_of_switching(
        self,
        treatment: str,
        outcome: str,
        low: float = 0.0,
        high: float = 1.0,
        n: int = 2_000_000,
        seed: int = 0,
    ) -> float:
        """E[outcome | do(treatment := high)] - E[outcome | do(treatment := low)].

        The effect to use for a binary treatment, where "shift everyone by +1" would
        push the already-treated to 2. Same paired draws as `true_effect`."""
        if outcome not in self.observed:
            raise KeyError(f"outcome {outcome!r} is not an observed node")
        lo = self.sample(n, seed, set_to={treatment: low})[outcome].to_numpy(dtype=float)
        hi = self.sample(n, seed, set_to={treatment: high})[outcome].to_numpy(dtype=float)
        return float(hi.mean() - lo.mean())

    def true_subgroup_effects(
        self,
        treatment: str,
        outcome: str,
        by: str,
        low: float = 0.0,
        high: float = 1.0,
        n: int = 2_000_000,
        seed: int = 0,
    ) -> dict[str, float]:
        """do(treatment := high) minus do(treatment := low), within each level of the
        binary node `by`: {"0": effect, "1": effect}.

        `by` must not be affected by `treatment` (a pre-treatment covariate), otherwise
        the subgroup a unit belongs to would change with the intervention and "the effect
        within a subgroup" would not be defined; that raises."""
        if not self.is_binary(by):
            raise ValueError(f"{by!r} is not a binary node")
        lo = self.sample(n, seed, set_to={treatment: low})
        hi = self.sample(n, seed, set_to={treatment: high})
        if not lo[by].equals(hi[by]):
            raise ValueError(f"{by!r} is affected by {treatment!r}; it is not a pre-treatment subgroup variable")
        y_lo, y_hi = lo[outcome].to_numpy(dtype=float), hi[outcome].to_numpy(dtype=float)
        groups = lo[by].to_numpy()
        return {
            str(level): float(y_hi[groups == level].mean() - y_lo[groups == level].mean())
            for level in (0, 1)
            if (groups == level).any()
        }
