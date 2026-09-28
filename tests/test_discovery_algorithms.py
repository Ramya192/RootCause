"""Stage 3 with PC, GES and LiNGAM (rootcause/pipeline/causal_discovery.py).

Real algorithms on simulated data with a known graph. The attrition SCM's graph is
compensation -> satisfaction <- manager_quality -> burnout <- workload, both -> attrition.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from rootcause.evaluation import harness, scms, stress
from rootcause.pipeline import causal_discovery

CONTINUOUS = ["compensation", "manager_quality", "workload", "job_satisfaction", "burnout"]


def _config(domain_config: dict, algorithm: str, variables=None, priors=True) -> dict:
    cfg = copy.deepcopy(domain_config)
    cfg["causal_discovery"]["algorithm"] = algorithm
    if variables is not None:
        cfg["causal_discovery"]["variables"] = variables
    if not priors:
        cfg["causal_discovery"]["required_edges"] = []
        cfg["causal_discovery"]["forbidden_edges"] = []
    return cfg


def _truth(scm, variables) -> set:
    return {e for e in scm.edges() if set(e) <= set(variables)}


# --- validation ---------------------------------------------------------------------------


def test_unknown_algorithm_raises_instead_of_silently_running_pc(feature_df, domain_config):
    with pytest.raises(ValueError, match="causal_discovery.algorithm='fci' is not supported"):
        causal_discovery.discover_graph(feature_df, _config(domain_config, "fci"))


def test_priors_naming_a_variable_outside_the_search_are_a_clear_error(feature_df, domain_config):
    cfg = _config(domain_config, "pc", variables=CONTINUOUS)  # the config's priors still mention attrition
    with pytest.raises(ValueError, match=r"not in causal_discovery.variables"):
        causal_discovery.discover_graph(feature_df, cfg)


def test_graph_records_the_algorithm_that_ran(feature_df, domain_config):
    for algorithm in causal_discovery.ALGORITHMS:
        assert causal_discovery.discover_graph(feature_df, _config(domain_config, algorithm)).algorithm == algorithm


# --- PC and GES agree where the structure is identifiable ----------------------------------


def test_ges_recovers_the_attrition_graph_and_honours_the_priors(feature_df, domain_config):
    graph = causal_discovery.discover_graph(feature_df, _config(domain_config, "ges"))
    assert set(graph.edges) == set(scms.attrition_scm().edges())
    forbidden = {tuple(e) for e in domain_config["causal_discovery"]["forbidden_edges"]}
    assert not set(graph.edges) & forbidden
    assert {tuple(e) for e in domain_config["causal_discovery"]["required_edges"]} <= set(graph.edges)


def test_ges_and_pc_recover_the_same_edges_without_priors_on_gaussian_noise(domain_config):
    df = scms.attrition_scm().sample(2000, 1001)
    found = {
        algo: set(causal_discovery.discover_graph(df, _config(domain_config, algo, priors=False)).edges)
        for algo in ("pc", "ges")
    }
    assert found["pc"] == found["ges"]  # same CPDAG on this draw (both drop what they cannot orient)
    assert found["ges"] <= set(scms.attrition_scm().edges())  # and nothing invented


# --- GES: priors are applied after the search ---------------------------------------------


def _two_variable_frame(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=3000)
    return pd.DataFrame({"a": a, "b": 0.8 * a + rng.normal(size=3000) * 0.6})  # a -> b is Markov-equivalent to b -> a


def _cfg2(**priors) -> dict:
    return {"causal_discovery": {"algorithm": "ges", "variables": ["a", "b"], **priors}}


def test_ges_leaves_a_markov_equivalent_pair_undirected_and_drops_it():
    assert causal_discovery.discover_graph(_two_variable_frame(), _cfg2()).edges == []


def test_ges_orients_an_undirected_edge_when_the_priors_forbid_one_direction():
    graph = causal_discovery.discover_graph(_two_variable_frame(), _cfg2(forbidden_edges=[["b", "a"]]))
    assert graph.edges == [("a", "b")]
    graph = causal_discovery.discover_graph(_two_variable_frame(), _cfg2(forbidden_edges=[["a", "b"]]))
    assert graph.edges == [("b", "a")]


def test_ges_drops_the_edge_when_the_priors_forbid_both_directions_and_adds_required_ones():
    both = _cfg2(forbidden_edges=[["a", "b"], ["b", "a"]])
    assert causal_discovery.discover_graph(_two_variable_frame(), both).edges == []
    required = _cfg2(required_edges=[["a", "b"]])
    assert causal_discovery.discover_graph(_two_variable_frame(), required).edges == [("a", "b")]


# --- LiNGAM -------------------------------------------------------------------------------


def test_lingam_recovers_the_continuous_graph_when_noise_is_non_gaussian(domain_config):
    """Linear, acyclic, uniform noise, continuous variables only: every LiNGAM assumption holds."""
    scm = scms.non_gaussian_attrition_scm()
    df = scm.sample(2000, 1000)
    cfg = _config(domain_config, "lingam", variables=CONTINUOUS, priors=False)
    found = set(causal_discovery.discover_graph(df, cfg).edges)
    assert found == _truth(scm, CONTINUOUS)


def test_lingam_cannot_orient_gaussian_data_and_invents_edges(domain_config):
    """The identification argument needs non-Gaussian noise: with Gaussian noise the directions are
    not identified, so wrong edges appear. This is a property of the method, not a bug."""
    scm = scms.attrition_scm()
    df = scm.sample(2000, 1001)
    cfg = _config(domain_config, "lingam", variables=CONTINUOUS, priors=False)
    found = set(causal_discovery.discover_graph(df, cfg).edges)
    assert len(found - _truth(scm, CONTINUOUS)) >= 2


def test_lingam_gets_forbidden_edges_as_no_path_prior_knowledge(monkeypatch):
    import lingam

    captured = {}

    class Spy:
        def __init__(self, prior_knowledge=None, **kwargs):
            captured["prior"] = prior_knowledge

        def fit(self, X):
            self.adjacency_matrix_ = np.zeros((X.shape[1], X.shape[1]))
            return self

    monkeypatch.setattr(lingam, "DirectLiNGAM", Spy)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(50, 3)), columns=["x", "y", "z"])
    cfg = {"causal_discovery": {"algorithm": "lingam", "variables": ["x", "y", "z"], "forbidden_edges": [["z", "x"]]}}

    causal_discovery.discover_graph(df, cfg)

    prior = captured["prior"]
    assert prior[0, 2] == 0  # row = child x, column = parent z: no directed path z -> x
    assert (prior == -1).sum() == 8  # everything else is unknown


def test_lingam_threshold_prunes_weak_edges(monkeypatch):
    import lingam

    class Fake:
        def __init__(self, **kwargs):
            pass

        def fit(self, X):
            self.adjacency_matrix_ = np.array([[0, 0.5, 0], [0, 0, 0.03], [0, 0, 0]])  # x1 <- x2 (0.5), x2 <- x3 (0.03)
            return self

    monkeypatch.setattr(lingam, "DirectLiNGAM", Fake)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 3)), columns=["x1", "x2", "x3"])
    base = {"causal_discovery": {"algorithm": "lingam", "variables": ["x1", "x2", "x3"]}}

    assert causal_discovery.discover_graph(df, base).edges == [("x2", "x1")]  # 0.03 < the 0.05 default
    loose = copy.deepcopy(base)
    loose["causal_discovery"]["lingam_threshold"] = 0.01
    assert causal_discovery.discover_graph(df, loose).edges == [("x2", "x1"), ("x3", "x2")]


def test_lingam_without_the_package_says_how_to_install_it(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name == "lingam":
            raise ImportError("no lingam")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    df = pd.DataFrame(np.random.default_rng(0).normal(size=(30, 2)), columns=["a", "b"])
    with pytest.raises(ImportError, match="pip install lingam"):
        causal_discovery.discover_graph(df, {"causal_discovery": {"algorithm": "lingam", "variables": ["a", "b"]}})


# --- the non-Gaussian stress scenarios ----------------------------------------------------


def test_non_gaussian_scm_keeps_the_baseline_graph_and_variances_but_not_the_noise_shape():
    base, uniform = scms.attrition_scm(), scms.non_gaussian_attrition_scm()
    assert set(uniform.edges()) == set(base.edges())
    b, u = base.sample(60_000, 1), uniform.sample(60_000, 1)
    for column in CONTINUOUS:
        assert u[column].std() == pytest.approx(b[column].std(), rel=0.03), column
    from scipy import stats

    assert abs(stats.kurtosis(b["compensation"])) < 0.15  # Gaussian: excess kurtosis 0
    assert stats.kurtosis(u["compensation"]) == pytest.approx(-1.2, abs=0.1)  # uniform: -1.2
    assert u["attrition"].mean() == pytest.approx(b["attrition"].mean(), abs=0.02)


def test_continuous_only_scenario_drops_the_outcome_and_its_priors_from_stage_3(domain_config):
    before = copy.deepcopy(domain_config)
    cfg = stress.discovery_without_outcome(domain_config)

    assert domain_config == before  # the shared config is not mutated
    assert "attrition" not in cfg["causal_discovery"]["variables"]
    for key in ("required_edges", "forbidden_edges"):
        assert all("attrition" not in edge for edge in cfg["causal_discovery"][key])
    assert cfg["effect_estimation"]["outcome"] == "attrition"  # Stages 4-6 still use it
    scenario = {s.id: s for s in stress.STRESS_SCENARIOS}["non_gaussian_continuous_only"]
    truth = harness.compute_truth(scenario.scm(), scenario.configure(domain_config))
    assert len(truth.edges) == 4 and all("attrition" not in e for e in truth.edges)


def test_with_algorithm_forces_stage_3_without_mutating_the_config(domain_config):
    forced = harness.with_algorithm(domain_config, "ges")
    assert forced["causal_discovery"]["algorithm"] == "ges"
    assert domain_config["causal_discovery"].get("algorithm", "pc") == "pc"
    assert harness.with_algorithm(domain_config, None) is domain_config
    assert forced["effect_estimation"] is domain_config["effect_estimation"]
