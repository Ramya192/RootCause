"""Causal anomaly flagging (rootcause/pipeline/anomalies.py) and its evaluation on injected faults."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from rootcause.evaluation import anomalies_eval
from rootcause.models.schemas import CausalGraph
from rootcause.pipeline import anomalies


def _chain(n=4000, seed=0) -> tuple[pd.DataFrame, CausalGraph]:
    """x -> y (linear, sd 0.5) -> z (logistic): one continuous mechanism, one binary."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.5, size=n)
    z = (rng.uniform(size=n) < 1 / (1 + np.exp(-2.0 * y))).astype(int)
    df = pd.DataFrame({"x": x, "y": y, "z": z})
    return df, CausalGraph(nodes=["x", "y", "z"], edges=[("x", "y"), ("y", "z")], algorithm="truth")


# --- mechanisms ---------------------------------------------------------------------------


def test_mechanisms_recover_the_generating_process():
    df, graph = _chain()
    mech = anomalies.fit_mechanisms(df, graph)

    assert mech["x"].parents == () and not mech["x"].binary
    assert mech["y"].parents == ("x",) and mech["y"].sigma == pytest.approx(0.5, abs=0.02)
    assert mech["y"].model.coef_[0] == pytest.approx(1.5, abs=0.03)
    assert mech["z"].binary and mech["z"].parents == ("y",)
    assert mech["z"].expected(pd.DataFrame({"y": [0.0]}))[0] == pytest.approx(0.5, abs=0.05)  # sigmoid(0)


def test_surprise_is_centred_so_a_clean_record_scores_about_zero():
    df, graph = _chain()
    surprises = anomalies.flag_anomalies(df, graph).surprises
    assert surprises.mean().abs().max() < 0.05  # every variable's surprise averages ~0 on data it was fitted on


def test_an_unexplained_binary_outcome_is_more_surprising_than_an_expected_one():
    df, graph = _chain()
    mech = anomalies.fit_mechanisms(df, graph)["z"]
    probe = pd.DataFrame({"y": [3.0, 3.0], "z": [1, 0]})  # y=3 makes z=1 near-certain
    surprise = mech.surprise(probe)
    assert surprise[1] > 3.0 > surprise[0]  # z=0 is very surprising; z=1 is expected (slightly below centre)


def test_missing_columns_and_an_unparented_graph_are_handled():
    df, graph = _chain()
    with pytest.raises(ValueError, match="not columns of the data"):
        anomalies.fit_mechanisms(df.drop(columns="y"), graph)
    flat = CausalGraph(nodes=["x", "y"], edges=[], algorithm="pc")
    report = anomalies.flag_anomalies(df, flat)
    assert set(report.surprises.columns) == {"x", "y"}  # marginal models when nothing is known about causes


# --- categorical / ordinal variables ---------------------------------------------------------


def _with_deductible(n=4000, seed=0):
    """x -> y (linear) plus a parentless 4-level 'deductible' that is 300 for 93% of records:
    near-constant, so a Gaussian sees every other level as many sd from the mean."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = 1.5 * x + rng.normal(scale=0.5, size=n)
    d = rng.choice([300, 400, 500, 700], size=n, p=[0.93, 0.04, 0.02, 0.01])
    df = pd.DataFrame({"x": x, "y": y, "d": d})
    graph = CausalGraph(nodes=["x", "y", "d"], edges=[("x", "y")], algorithm="truth")
    return df, graph


def test_variable_kind_by_number_of_levels():
    df, _ = _with_deductible(500)
    assert anomalies.variable_kind(df["d"]) == "categorical"
    assert anomalies.variable_kind(df["y"]) == "continuous"
    assert anomalies.variable_kind(pd.Series([0, 1, 1, 0])) == "binary"
    assert anomalies.variable_kind(df["d"], max_levels=None) == "continuous"  # the earlier Gaussian treatment
    assert anomalies.variable_kind(pd.Series(range(13)), max_levels=12) == "continuous"


def test_a_categorical_mechanism_recovers_the_conditional_level_frequencies():
    rng = np.random.default_rng(3)
    n = 6000
    x = rng.integers(0, 2, size=n)  # binary parent
    level = np.where(x == 1, rng.choice([1, 2, 3], size=n, p=[0.1, 0.2, 0.7]), rng.choice([1, 2, 3], size=n, p=[0.6, 0.3, 0.1]))
    df = pd.DataFrame({"x": x, "c": level})
    graph = CausalGraph(nodes=["x", "c"], edges=[("x", "c")], algorithm="truth")
    mech = anomalies.fit_mechanisms(df, graph)["c"]

    assert mech.kind == "categorical" and mech.classes == (1.0, 2.0, 3.0)
    probe = pd.DataFrame({"x": [0, 1], "c": [1, 3]})
    assert mech.probability(probe) == pytest.approx([0.6, 0.7], abs=0.03)
    assert list(mech.expected(probe)) == [1.0, 3.0]  # the most likely level given x
    surprises = anomalies.flag_anomalies(df, graph).surprises["c"]
    assert abs(surprises.mean()) < 0.03  # centred on data it was fitted on, like the other mechanisms


def test_an_unseen_level_gets_the_probability_floor():
    df, graph = _with_deductible()
    mech = anomalies.fit_mechanisms(df, graph)["d"]
    probe = pd.DataFrame({"d": [300, 999]})
    prob = mech.probability(probe)
    assert prob[0] == pytest.approx(0.93, abs=0.02) and prob[1] == anomalies.PROBABILITY_FLOOR


def test_a_single_level_variable_does_not_crash_and_is_never_surprising():
    df, graph = _with_deductible(300)
    df["k"] = 5
    graph = CausalGraph(nodes=["x", "y", "d", "k"], edges=[("x", "y"), ("x", "k")], algorithm="truth")
    report = anomalies.flag_anomalies(df, graph)
    assert report.surprises["k"].abs().max() < 0.01


def test_a_rare_level_no_longer_dominates_the_ranking_as_it_did_under_a_gaussian_model():
    df, graph = _with_deductible()
    corrupted, mask = _swap_y(df, 120)  # 3% of records get a y that does not fit their x
    earlier = anomalies.flag_anomalies(corrupted, graph, max_levels=None, score_roots=True)
    typed_only = anomalies.flag_anomalies(corrupted, graph, score_roots=True)  # categorical, roots still scored
    default = anomalies.flag_anomalies(corrupted, graph)

    rare = (corrupted["d"] == 700).to_numpy()  # ~1% of records
    share = lambda r: rare[np.argsort(-r.scores.to_numpy())[:100]].mean()  # noqa: E731
    assert share(earlier) > 0.4  # the reported failure: the top of the list is the 700 level
    assert share(typed_only) < share(earlier) - 0.15  # a categorical model helps, but rarity itself is still surprising
    assert share(default) < 0.05  # a root has no mechanism to break: not scored

    top_default = np.argsort(-default.scores.to_numpy())[:100]
    assert mask[top_default].mean() > 0.5  # what is left at the top is the actual faults
    assert roc_auc_score(mask, default.scores) >= roc_auc_score(mask, earlier.scores)


def test_roots_are_reported_but_not_scored_unless_asked_and_a_flat_graph_scores_everything():
    df, graph = _with_deductible()
    report = anomalies.flag_anomalies(df, graph)
    assert set(report.surprises.columns) == {"x", "y", "d"}  # every variable's surprise stays visible
    assert report.scored == ("y",)  # only y has a parent
    assert (report.scores == report.surprises["y"]).all()
    assert set(report.top_variable) == {"y"}
    assert set(anomalies.flag_anomalies(df, graph, score_roots=True).scored) == {"x", "y", "d"}

    flat = CausalGraph(nodes=["x", "y"], edges=[], algorithm="pc")
    assert set(anomalies.flag_anomalies(df, flat).scored) == {"x", "y"}  # nothing else to score


def test_top_reports_the_probability_of_a_discrete_value_and_nan_for_a_continuous_one():
    df, graph = _with_deductible()
    top = anomalies.flag_anomalies(df, graph, score_roots=True).top(300)
    discrete = top[top.variable == "d"]
    assert len(discrete) and discrete["probability"].between(0, 0.05).all()  # rare levels: that is why they surfaced
    assert top[top.variable == "y"]["probability"].isna().all()


# --- detection ----------------------------------------------------------------------------


def _swap_y(df: pd.DataFrame, n_bad: int, seed=1) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    bad = rng.choice(len(df), n_bad, replace=False)
    out = df.copy()
    out.loc[out.index[bad], "y"] = out["y"].to_numpy()[rng.permutation(len(df))[:n_bad]]
    mask = np.zeros(len(df), dtype=bool)
    mask[bad] = True
    return out, mask


def test_a_swapped_value_is_invisible_to_a_marginal_check_but_flagged_causally():
    df, graph = _chain()
    corrupted, mask = _swap_y(df, 120)
    report = anomalies.flag_anomalies(corrupted, graph)
    marginal = np.abs((corrupted - corrupted.mean()) / corrupted.std()).max(axis=1)

    assert roc_auc_score(mask, report.scores) > 0.8  # not 1.0: a donor value near the original is a fault too small to see
    assert roc_auc_score(mask, marginal) < 0.65  # the swapped values are ordinary values: nothing marginal to see
    top = report.top_variable.to_numpy()[mask]
    assert np.mean(top == "y") > 0.5  # y itself, or its child z: the family


def test_a_clean_reference_makes_a_heavily_corrupted_set_easier_to_flag():
    df, graph = _chain()
    corrupted, mask = _swap_y(df, 900)  # 22% corrupted: enough to bend the mechanism it is fitted on
    contaminated = roc_auc_score(mask, anomalies.flag_anomalies(corrupted, graph).scores)
    clean_fit = roc_auc_score(mask, anomalies.flag_anomalies(corrupted, graph, fit_on=df).scores)
    assert clean_fit > contaminated


def test_top_lists_the_most_surprising_records_with_observed_and_expected_values():
    df, graph = _chain()
    corrupted, mask = _swap_y(df, 60)
    top = anomalies.flag_anomalies(corrupted, graph).top(10)

    assert list(top.columns) == ["record", "score", "variable", "observed", "expected", "surprise", "probability"]
    assert len(top) == 10 and top["score"].is_monotonic_decreasing
    assert mask[top["record"].to_numpy()].mean() > 0.7  # the head of the list is mostly the corrupted records
    row = top.iloc[0]
    assert abs(row["observed"] - row["expected"]) > 0 and row["surprise"] > 0


def test_records_are_indexed_by_the_entity_id_when_the_config_names_one():
    df, graph = _chain(n=500)
    df.insert(0, "claim_id", range(1001, 1501))
    cfg = {"feature_store": {"entity_id_column": "claim_id"}}
    report = anomalies.flag_anomalies(df, graph, cfg)
    assert list(report.scores.index[:3]) == [1001, 1002, 1003]
    assert 1500 in report.top(500)["record"].to_numpy()


# --- the evaluation -----------------------------------------------------------------------


def test_inject_swap_uses_existing_values_and_shift_moves_by_three_sd():
    df, _ = _chain(n=1000)
    swapped, mask, victim = anomalies_eval.inject(df, ["y"], "swap", 0.05, seed=3)
    assert mask.sum() == 50 and (victim[mask] == "y").all() and (victim[~mask] == "").all()
    assert set(swapped["y"].round(9)) <= set(df["y"].round(9))  # every value already existed
    assert (swapped.loc[~mask] == df.loc[~mask]).all().all()  # clean rows untouched
    assert (swapped.loc[mask, "y"] != df.loc[mask, "y"]).all()

    shifted, mask, _ = anomalies_eval.inject(df, ["y"], "shift", 0.05, seed=3)
    assert np.allclose(shifted.loc[mask, "y"] - df.loc[mask, "y"], 3.0 * df["y"].std())

    flipped, mask, _ = anomalies_eval.inject(df, ["z"], "shift", 0.05, seed=3)
    assert (flipped.loc[mask, "z"] == 1 - df.loc[mask, "z"]).all()  # binary: flipped
    swapped_z, mask, _ = anomalies_eval.inject(df, ["z"], "swap", 0.05, seed=3)
    assert (swapped_z.loc[mask, "z"] != df.loc[mask, "z"]).all()  # a swap that changes nothing would not be a fault


def test_inject_is_seeded_and_rounds_the_count_up():
    df, _ = _chain(n=101)
    a, mask_a, _ = anomalies_eval.inject(df, ["y"], "swap", 0.03, seed=9)
    b, mask_b, _ = anomalies_eval.inject(df, ["y"], "swap", 0.03, seed=9)
    pd.testing.assert_frame_equal(a, b)
    assert mask_a.sum() == 4 == mask_b.sum()  # ceil(0.03 * 101)


def test_two_draws_end_to_end_causal_flagging_finds_shifts_and_names_the_variable(monkeypatch):
    monkeypatch.setattr(anomalies_eval, "SCENARIOS", anomalies_eval.SCENARIOS[:1])  # attrition only
    result = anomalies_eval.run(replicates=2)
    df = pd.DataFrame([r.__dict__ for r in result["rows"]])

    assert set(df.method) == set(anomalies_eval.METHODS)
    assert set(zip(df.fault, df.target_kind)) == {
        (f, k) for f in ("swap", "shift") for k in ("continuous", "binary", "root")
    }
    auc = df.groupby(["fault", "target_kind", "method"]).roc_auc.mean()
    # a plain outlier in a continuous variable: found, and the right variable is named
    assert auc["shift", "continuous", "causal (true graph)"] > 0.95
    top1 = df[(df.fault == "shift") & (df.target_kind == "continuous") & (df.method == "causal (true graph)")].top1_attribution
    assert top1.mean() > 0.9
    # a swapped value: ordinary on its own, so marginal checks are near chance; causal flagging is well above
    assert auc["swap", "continuous", "marginal z-score"] < 0.6
    assert auc["swap", "continuous", "causal (true graph)"] > auc["swap", "continuous", "marginal z-score"] + 0.08
    assert auc["swap", "continuous", "causal (true graph)"] > auc["swap", "continuous", "isolation forest"]
    # flipping a binary outcome to a plausible value is close to invisible to every method: say so, don't hide it
    assert auc["shift", "binary", "causal (true graph)"] < 0.75
    causal = df[df.method.str.startswith("causal")]
    assert causal.top1_attribution.notna().all() and df[~df.method.str.startswith("causal")].top1_attribution.isna().all()
    assert (causal.family_attribution >= causal.top1_attribution).all()

    md = anomalies_eval.render_markdown(result)
    assert "## attrition: swap in a continuous variable" in md and "## attrition: shift in a binary variable" in md
    assert "Mahalanobis" in md and "nan" not in md and "Binary variables are hard" in md
    assert json.loads(anomalies_eval.render_json(result))["settings"]["replicates"] == 2
