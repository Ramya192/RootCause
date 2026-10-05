"""Pure scoring functions: pipeline output + ground truth -> numbers.

Nothing here runs the pipeline or touches data, so each metric can be tested
against hand-computed cases. A metric that is undefined (e.g. precision when
no edges were predicted) is None, never a fake 0 or 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from scipy.stats import kendalltau

from causal_engine.models.schemas import EffectEstimate, InterventionRecommendation

Edge = tuple[str, str]


def _ratio(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def structural_hamming_distance(predicted: Iterable[Edge], true: Iterable[Edge]) -> int:
    """Edge edits (add, delete or reverse) needed to turn `predicted` into `true`.

    Each unordered node pair is in one of three states: no edge, a->b, b->a.
    The distance counts pairs whose state differs, so a reversed edge costs 1
    (not 2), and an edge Stage 3 left unoriented and dropped counts as a miss.
    """
    predicted, true = set(predicted), set(true)
    pairs = {frozenset(e) for e in predicted | true}

    def state(edges: set[Edge], pair: frozenset) -> Optional[Edge]:
        a, b = sorted(pair)
        if (a, b) in edges:
            return (a, b)
        if (b, a) in edges:
            return (b, a)
        return None

    return sum(state(predicted, p) != state(true, p) for p in pairs)


@dataclass(frozen=True)
class GraphScore:
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    shd: int
    true_edges_found: int
    true_edges: int
    # Same idea, ignoring edges the domain config supplied as priors
    # (`required_edges`): those are asserted, not discovered, so they flatter
    # both precision and recall. This is the honest measure of what PC found.
    discovered_precision: Optional[float]
    discovered_recall: Optional[float]


def score_graph(
    predicted: Iterable[Edge], true: Iterable[Edge], priors: Iterable[Edge] = ()
) -> GraphScore:
    predicted, true, priors = set(predicted), set(true), set(priors)
    tp = len(predicted & true)
    precision, recall = _ratio(tp, len(predicted)), _ratio(tp, len(true))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )

    free_pred, free_true = predicted - priors, true - priors
    return GraphScore(
        precision=precision,
        recall=recall,
        f1=f1,
        shd=structural_hamming_distance(predicted, true),
        true_edges_found=tp,
        true_edges=len(true),
        discovered_precision=_ratio(len(free_pred & free_true), len(free_pred)),
        discovered_recall=_ratio(len(free_pred & free_true), len(free_true)),
    )


@dataclass(frozen=True)
class EffectRow:
    treatment: str
    estimated: float
    true: float

    @property
    def abs_error(self) -> float:
        return abs(self.estimated - self.true)

    @property
    def sign_agrees(self) -> bool:
        # Signs are compared as -1/0/+1, so an estimate of exactly 0 has no sign and does
        # not count as agreeing with a negative truth. (Not meaningful when the truth is 0.)
        return (self.estimated > 0) - (self.estimated < 0) == (self.true > 0) - (self.true < 0)


def score_effects(
    estimates: Iterable[EffectEstimate], true_effects: Mapping[str, float]
) -> list[EffectRow]:
    """One row per estimated treatment that has a known true effect."""
    return [
        EffectRow(treatment=e.treatment, estimated=e.ate, true=true_effects[e.treatment])
        for e in estimates
        if e.treatment in true_effects
    ]


def refutation_pass_rate(estimates: Iterable[EffectEstimate]) -> Optional[float]:
    """Share of estimates whose refuter passed; estimates with no refuter are skipped."""
    outcomes = [e.refutation_passed for e in estimates if e.refutation_passed is not None]
    return _ratio(sum(outcomes), len(outcomes))


@dataclass(frozen=True)
class RankingScore:
    chosen: str  # the pipeline's rank-1 intervention
    true_best: str  # best by true ROI among the candidates the pipeline ranked
    top1_correct: bool
    kendall_tau: Optional[float]  # rank agreement over all ranked candidates; None if < 2
    roi_regret: float  # true ROI of the true best minus true ROI of the one chosen (>= 0)


def score_ranking(
    recommendations: Iterable[InterventionRecommendation], true_roi: Mapping[str, float]
) -> Optional[RankingScore]:
    """Compare the pipeline's ranking to the ranking by TRUE ROI.

    Only candidates the pipeline ranked and that have a true ROI are compared.
    Returns None if there are none.
    """
    ranked = sorted(
        (r for r in recommendations if r.id in true_roi), key=lambda r: r.rank
    )
    if not ranked:
        return None

    chosen = ranked[0].id
    true_best = max((r.id for r in ranked), key=lambda cid: true_roi[cid])

    tau = None
    if len(ranked) >= 2:
        # rank 1 = best, so negate rank to correlate "higher = better" on both sides
        tau_value = kendalltau(
            [-r.rank for r in ranked], [true_roi[r.id] for r in ranked]
        ).statistic
        tau = None if tau_value != tau_value else float(tau_value)  # NaN (all ties) -> None

    return RankingScore(
        chosen=chosen,
        true_best=true_best,
        top1_correct=chosen == true_best,
        kendall_tau=tau,
        roi_regret=true_roi[true_best] - true_roi[chosen],
    )
