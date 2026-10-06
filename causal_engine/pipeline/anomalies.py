"""Causal anomaly flagging: records whose values are surprising GIVEN their causes.

An ordinary outlier detector asks "is this record unusual?" and looks at each variable, or
their joint distribution, without a notion of cause. It misses the interesting case: a
record whose every value is common, but whose combination breaks the mechanism -- an
employee with terrible workload, poor management and low pay who is nonetheless highly
satisfied, or a claim whose fraud outcome nothing in its causes explains.

This module fits a mechanism P(variable | its parents) for every variable in the Stage 3
graph and scores each record by how surprising its values are under those mechanisms. The
mechanism follows the variable's TYPE:

  * binary (0/1)                      logistic regression
  * categorical / ordinal (a non-binary column with at most `max_levels` distinct values:
    a deductible of 300/400/500/700, a 1-5 age band)   multinomial logistic regression
  * anything else (continuous)        linear regression with Gaussian noise
  * a variable with no parents gets its marginal (frequencies, or a Gaussian)

Modelling a discrete column as Gaussian is the mistake to avoid: a 4-level deductible with
a mode of 300 looks like a continuous variable with a tiny sd, so every rarer level is
"many sd from the mean" and dominates the ranking without being anomalous in any useful way.
The score is then, for every variable,

    surprise(v) = -log P(x_v | parents)  -  its expected value under the fitted model

so every variable's surprise is centred at zero for a typical record and the sum over
variables is comparable across records. The record's score is that sum; the variable
with the largest surprise is reported as the mechanism most likely broken.

Limits, stated so the output is not over-read:
  * It is only as good as the graph and the mechanisms: a missing parent makes typical
    records look surprising; a wrong functional form does too.
  * A fault in one variable also makes its CHILDREN look surprising (their parent's value
    changed), so the top variable is the fault or one of its children. It is a lead for an
    investigator, not a root-cause proof.
  * Flagged does not mean wrong or fraudulent: it means "not explained by the model".
  * A variable with no parents (a root) has no mechanism to break: its "surprise" is only
    how rare its value is, which is what a marginal outlier check already measures. A 1%
    category would fill the top of the list on rarity alone. So by default the record score
    sums the surprises of variables that HAVE parents in the graph; a fault in a root still
    shows up through its children. Pass `score_roots=True` to include the roots (a graph
    with no edges scores every variable, as there is nothing else to score).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from causal_engine.models.schemas import CausalGraph

PROBABILITY_FLOOR = 1e-6
MIN_SIGMA = 1e-9
MAX_LEVELS = 12  # a non-binary column with at most this many distinct values is categorical
SMOOTHING = 0.5  # pseudo-count per level for a parentless categorical variable


@dataclass(frozen=True)
class Mechanism:
    variable: str
    parents: tuple[str, ...]
    kind: str  # "continuous" | "binary" | "categorical"
    model: object  # fitted regressor / classifier, or None for a parentless variable
    constant: float  # marginal mean (continuous) or P(x=1) (binary) when there are no parents
    sigma: float  # residual sd (continuous only)
    classes: tuple[float, ...] = ()  # categorical only: the levels, ascending
    marginal: tuple[float, ...] = ()  # categorical, no parents: smoothed level frequencies

    @property
    def binary(self) -> bool:
        return self.kind == "binary"

    def _class_probs(self, df: pd.DataFrame) -> np.ndarray:
        """(records x levels) probabilities of a categorical variable, columns in `classes` order."""
        if self.model is None:  # no parents, or a single level (nothing to regress on)
            return np.tile(np.asarray(self.marginal), (len(df), 1))
        return self.model.predict_proba(df[list(self.parents)].to_numpy(dtype=float))

    def expected(self, df: pd.DataFrame) -> np.ndarray:
        """E[x_v | parents] for a continuous variable, P(x_v = 1 | parents) for a binary one,
        the most likely level for a categorical one."""
        if self.kind == "categorical":
            return np.asarray(self.classes)[self._class_probs(df).argmax(axis=1)]
        if not self.parents:
            return np.full(len(df), self.constant)
        X = df[list(self.parents)].to_numpy(dtype=float)
        if self.binary:
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

    def probability(self, df: pd.DataFrame) -> np.ndarray:
        """P(the observed value | parents) for a discrete variable; NaN for a continuous one."""
        x = df[self.variable].to_numpy(dtype=float)
        if self.kind == "continuous":
            return np.full(len(df), np.nan)
        if self.binary:
            mu = self.expected(df)
            return np.clip(np.where(x == 1, mu, 1 - mu), PROBABILITY_FLOOR, 1.0)
        probs = self._class_probs(df)
        levels = np.asarray(self.classes)
        idx = np.abs(x[:, None] - levels[None, :]).argmin(axis=1)
        seen = np.isclose(levels[idx], x)
        return np.where(seen, np.clip(probs[np.arange(len(x)), idx], PROBABILITY_FLOOR, 1.0), PROBABILITY_FLOOR)

    def surprise(self, df: pd.DataFrame) -> np.ndarray:
        """Centred negative log-likelihood of each record's value (0 for a typical record)."""
        x = df[self.variable].to_numpy(dtype=float)
        if self.kind == "categorical":
            probs = np.clip(self._class_probs(df), PROBABILITY_FLOOR, 1.0)
            entropy = -(probs * np.log(probs)).sum(axis=1)
            return -np.log(self.probability(df)) - entropy
        mu = self.expected(df)
        if self.binary:
            p = np.clip(mu, PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)
            nll = -(x * np.log(p) + (1 - x) * np.log(1 - p))
            entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
            return nll - entropy
        z = (x - mu) / self.sigma
        return 0.5 * z**2 - 0.5


def _is_binary(series: pd.Series) -> bool:
    return set(pd.unique(series.dropna())) <= {0, 1}


def variable_kind(series: pd.Series, max_levels: Optional[int] = MAX_LEVELS) -> str:
    """'binary', 'categorical' (few distinct values), or 'continuous'. `max_levels` of None or 0
    turns categorical off, which reproduces the earlier Gaussian treatment of every non-binary column."""
    if _is_binary(series):
        return "binary"
    if max_levels and series.nunique(dropna=True) <= max_levels:
        return "categorical"
    return "continuous"


def fit_mechanisms(
    df: pd.DataFrame, graph: CausalGraph, max_levels: Optional[int] = MAX_LEVELS
) -> dict[str, Mechanism]:
    """One mechanism per node of `graph`, each fitted on `df` given the node's parents in the graph."""
    parents_of = {v: tuple(p for p, c in graph.edges if c == v) for v in graph.nodes}
    missing = [v for v in graph.nodes if v not in df.columns]
    if missing:
        raise ValueError(f"graph variables {missing} are not columns of the data")
    mechanisms = {}
    for variable in graph.nodes:
        parents = parents_of[variable]
        y = df[variable].to_numpy(dtype=float)
        kind = variable_kind(df[variable], max_levels)
        if kind == "categorical":
            levels, counts = np.unique(y, return_counts=True)
            marginal = (counts + SMOOTHING) / (counts.sum() + SMOOTHING * len(levels))
            model = None
            if parents and len(levels) > 1:
                X = df[list(parents)].to_numpy(dtype=float)
                model = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)).fit(X, y)
            mechanisms[variable] = Mechanism(
                variable, parents, kind, model, float(np.mean(y)), 0.0,
                classes=tuple(float(v) for v in levels), marginal=tuple(float(v) for v in marginal),
            )
        elif not parents:
            sigma = float(max(np.std(y, ddof=1), MIN_SIGMA)) if kind == "continuous" else 0.0
            mechanisms[variable] = Mechanism(variable, parents, kind, None, float(np.mean(y)), sigma)
        else:
            X = df[list(parents)].to_numpy(dtype=float)
            if kind == "binary":
                model = make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=1000)).fit(X, y)
                mechanisms[variable] = Mechanism(variable, parents, kind, model, float(np.mean(y)), 0.0)
            else:
                model = LinearRegression().fit(X, y)
                resid = y - model.predict(X)
                sigma = float(max(np.sqrt(np.sum(resid**2) / max(len(y) - len(parents) - 1, 1)), MIN_SIGMA))
                mechanisms[variable] = Mechanism(variable, parents, kind, model, float(np.mean(y)), sigma)
    return mechanisms


@dataclass
class AnomalyReport:
    scores: pd.Series  # sum of centred surprises per record (higher = less explained)
    surprises: pd.DataFrame  # records x ALL variables (roots included, even when not scored)
    mechanisms: dict[str, Mechanism]
    graph: CausalGraph
    data: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    scored: tuple[str, ...] = ()  # the variables `scores` sums over

    @property
    def top_variable(self) -> pd.Series:
        """The scored variable whose mechanism each record breaks most."""
        return self.surprises[list(self.scored)].idxmax(axis=1)

    def top(self, k: int = 20) -> pd.DataFrame:
        """The `k` most surprising records: score, the variable to look at, and what it was against
        what its causes predicted (a probability for a binary variable, the most likely level for a
        categorical one), plus `probability`: P(observed value | causes) for a discrete variable, NaN
        for a continuous one."""
        order = self.scores.sort_values(ascending=False).head(k).index
        rows = []
        for record in order:
            variable = self.top_variable.loc[record]
            mech = self.mechanisms[variable]
            rows.append(
                {
                    "record": record,
                    "score": float(self.scores.loc[record]),
                    "variable": variable,
                    "observed": float(self.data.loc[record, variable]),
                    "expected": float(mech.expected(self.data.loc[[record]])[0]),
                    "surprise": float(self.surprises.loc[record, variable]),
                    "probability": float(mech.probability(self.data.loc[[record]])[0]),
                }
            )
        return pd.DataFrame(rows)


def flag_anomalies(
    feature_df: pd.DataFrame,
    graph: CausalGraph,
    domain_config: Optional[dict] = None,
    fit_on: Optional[pd.DataFrame] = None,
    max_levels: Optional[int] = MAX_LEVELS,
    score_roots: bool = False,
) -> AnomalyReport:
    """Score every record of `feature_df` against mechanisms fitted on `fit_on` (default:
    `feature_df` itself, which lets a strong anomaly pull its own mechanism a little; pass a
    clean reference set when you have one). Records are indexed by the entity id when the
    domain config names one. Non-binary columns with at most `max_levels` distinct values get a
    categorical mechanism; `max_levels=None` models them as Gaussian, and `score_roots=True` includes
    variables without parents in the score (both together are the earlier behaviour)."""
    frame = feature_df
    if domain_config is not None:
        id_col = domain_config.get("feature_store", {}).get("entity_id_column")
        if id_col in frame.columns:
            frame = frame.set_index(id_col, drop=False)
    variables = list(graph.nodes)
    data = frame[variables]
    reference = fit_on[variables] if fit_on is not None else data
    mechanisms = fit_mechanisms(reference, graph, max_levels)
    surprises = pd.DataFrame({v: mechanisms[v].surprise(data) for v in variables}, index=data.index)
    has_parents = {c for _, c in graph.edges}
    scored = variables if score_roots or not graph.edges else [v for v in variables if v in has_parents]
    return AnomalyReport(
        scores=surprises[scored].sum(axis=1),
        surprises=surprises,
        mechanisms=mechanisms,
        graph=graph,
        data=data,
        scored=tuple(scored),
    )
