"""Stage 7 narrative tiers (causal_engine/pipeline/narrative.py + explanation.py).

No test touches the network: the AutoGen chain runs against AutoGen's replay client and
the single-call tier against a stub, so the suite stays free and hermetic.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from causal_engine.models.schemas import EffectEstimate, InterventionRecommendation
from causal_engine.pipeline import counterfactuals, explanation, interventions, narrative

FACTS = (
    "Top-line drivers of attrition, ranked by causal effect size:\n"
    "- compensation: ATE=-0.0248 (backdoor.linear_regression)\n"
    "- workload: ATE=+0.0092 (backdoor.linear_regression) -- placebo check FAILED "
    "(permutation p=0.891): no evidence of an effect beyond noise\n"
    "Recommended action: 'raise_pay' (ROI=12.5, cost=$40,000) -- FAIRNESS FLAG: "
    "population-level disparity detected (group ratio 0.69)"
)
CAVEATS = dict(failed_refutation=True, fairness_flagged=True)
GOOD = (
    "Raising pay lowers attrition by about 2.5 percentage points. Workload shows no "
    "evidence of an effect beyond noise. The $40,000 pay plan has an ROI of 12.5 but "
    "raises a fairness concern (ratio 0.69)."
)


# --- number extraction and grounding ---


def test_extract_numbers_skips_identifiers_and_keeps_formatting():
    found = {raw: (v, d) for raw, v, d in narrative.extract_numbers(
        "deductible_100usd and terminated_0119 vs $1,200, 12.5% and -0.0248"
    )}
    assert set(found) == {"$1,200", "12.5%", "-0.0248"}
    assert found["$1,200"] == (1200.0, 0)
    assert found["-0.0248"] == (-0.0248, 4)


def test_good_narrative_passes():
    report = narrative.check_grounding(GOOD, FACTS, **CAVEATS)
    assert report.passed, report.detail()


@pytest.mark.parametrize("phrase", ["2.5 percentage points", "2.48 points", "about 2%"])
def test_percentage_forms_of_a_fact_are_grounded(phrase):
    text = f"Raising pay lowers attrition by {phrase}."
    assert narrative.check_grounding(text, FACTS).passed


def test_invented_number_is_rejected():
    report = narrative.check_grounding("Raising pay cuts attrition by 7.9 points.", FACTS)
    assert not report.passed
    assert report.ungrounded_numbers == ["7.9"]


def test_rounding_is_limited_to_the_precision_shown():
    # 2.48 is a fact; "3%" is not a rounding of it
    assert not narrative.check_grounding("about 3% fewer leavers", FACTS).passed


def test_failed_placebo_needs_a_no_evidence_caveat():
    text = "Workload raises attrition by 0.9 points and pay lowers it."
    report = narrative.check_grounding(text, FACTS, failed_refutation=True)
    assert not report.passed
    assert any("placebo" in c for c in report.missing_caveats)
    assert narrative.check_grounding(GOOD, FACTS, failed_refutation=True).passed


def test_fairness_flag_must_be_mentioned():
    text = "Raise pay: ROI of 12.5."
    assert not narrative.check_grounding(text, FACTS, fairness_flagged=True).passed
    assert narrative.check_grounding(text + " Watch for bias.", FACTS, fairness_flagged=True).passed


@pytest.mark.parametrize(
    "phrase", ["an ATE of 2.5", "the average treatment effect", "a statistically significant drop", "SHAP says", "the placebo test"]
)
def test_jargon_is_rejected_only_when_plain_language_is_required(phrase):
    text = f"Raising pay has {phrase} points."
    assert narrative.check_grounding(text, FACTS).passed  # the template is exempt by default
    report = narrative.check_grounding(text, FACTS, plain_language=True)
    assert not report.passed and report.jargon


def test_plain_language_narrative_passes_the_jargon_check():
    assert narrative.check_grounding(GOOD, FACTS, plain_language=True, **CAVEATS).passed


def test_numbers_inside_column_names_in_the_facts_are_quotable():
    # live-run false positive: facts said 'age37_49', the model wrote "ages 37 to 49"
    facts = "SHAP attribution agrees 'age37_49' carries the most weight."
    assert narrative.check_grounding("Employees aged 37 to 49 matter most.", facts).passed
    assert not narrative.check_grounding("Employees aged 38 to 49 matter most.", facts).passed


def test_scientific_notation_in_the_facts_matches_its_decimal_form():
    # live-run false positive: facts said ROI=-1.33e-05, the model wrote -0.0000133
    facts = "Recommended action: 'x' (ROI=-1.33e-05, cost=$152)"
    assert narrative.check_grounding("The return is about -0.0000133 per dollar at a cost of $152.", facts).passed
    assert not narrative.check_grounding("The return is about 0.0000150 per dollar.", facts).passed
    assert not narrative.check_grounding("The return is about 0.5 per dollar.", facts).passed  # not a loose 5e-3 tolerance


def test_not_statistically_significant_is_a_hedge_not_jargon():
    text = "The program's effect is not statistically significant, so there is no evidence it works."
    assert narrative.check_grounding(text, FACTS, plain_language=True, failed_refutation=True).passed
    assert narrative.check_grounding("This is a statistically significant effect.", FACTS, plain_language=True).jargon
    # plain-English 'significant' (no significant effect, a significant drop) is not jargon
    assert narrative.check_grounding("There is no significant effect of workload.", FACTS, plain_language=True).passed
    assert narrative.check_grounding("A significant drop of 2.5 points.", FACTS, plain_language=True).passed


def test_empty_narrative_fails():
    assert not narrative.check_grounding("  ", FACTS).passed


def test_template_passes_its_own_grounding(
    feature_df, effect_estimates, raw_df, domain_config
):
    cf = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)
    facts = explanation._template_narrative(domain_config, effect_estimates, cf, recs, {"x": 0.1})
    assert explanation._grounding(facts, facts, effect_estimates, recs).passed


def test_template_states_failed_placebo_and_fairness():
    estimates = [
        EffectEstimate(
            treatment="t", outcome="y", ate=0.002, estimator="e",
            refutation_passed=False, refutation_p_value=0.891,
        )
    ]
    recs = [
        InterventionRecommendation(
            id="r", target_variable="t", expected_effect=0.1, cost=5.0, roi=1.0,
            fairness_ratio=0.69, fairness_pass=False, rank=1,
        )
    ]
    text = explanation._template_narrative(
        {"domain": {"id": "d"}, "ingestion": {"outcome_column": "y"}}, estimates, [], recs, {"a": 1.0}
    )
    assert "no evidence of an effect beyond noise" in text and "p=0.891" in text
    assert "FAIRNESS FLAG" in text and "0.69" in text
    assert explanation._grounding(text, text, estimates, recs).passed


# --- the AutoGen chain ---

replay = pytest.importorskip("autogen_ext.models.replay")


def _chain(replies, **kw):
    client = replay.ReplayChatCompletionClient(replies)
    text = narrative.autogen_narrative(
        FACTS, model="gpt-4o-mini", audience="HR business partner", outcome="attrition",
        entities="employees", model_client=client, **kw,
    )
    return text, client


def test_chain_returns_the_writers_message_after_exactly_three_turns():
    text, client = _chain(["analyst draft", "No issues", GOOD, "SHOULD NEVER BE USED"])
    assert text == GOOD
    assert client._current_index == 3  # analyst, skeptic, writer -- a hard cap, not a loop


def test_chain_works_when_called_inside_a_running_event_loop():
    async def inside_loop():
        return _chain(["draft", "No issues", GOOD])[0]

    assert asyncio.run(inside_loop()) == GOOD


def test_chain_rejects_an_empty_writer_message():
    with pytest.raises(Exception):
        _chain(["draft", "No issues", "   "])


# --- tier selection ---


@pytest.fixture
def stage7(feature_df, effect_estimates, raw_df, domain_config, monkeypatch):
    """Run Stage 7 with an API key 'set' and the network tiers replaced by stubs."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    cf = counterfactuals.estimate_counterfactuals(feature_df, domain_config)
    recs = interventions.rank_interventions(raw_df, effect_estimates, domain_config)

    def run(autogen=None, llm=None, config_update=None):
        if autogen is not None:
            monkeypatch.setattr(narrative, "autogen_narrative", autogen)
        if llm is not None:
            monkeypatch.setattr(explanation, "_llm_narrative", llm)
        cfg = {**domain_config, "explanation": {**domain_config["explanation"], **(config_update or {})}}
        return explanation.generate_explanation(feature_df, effect_estimates, cf, recs, cfg)

    return run


def _grounded_text(facts_first_line_only=True):
    return "Attrition drivers were ranked by their causal effect."


def _boom(*a, **k):
    raise RuntimeError("simulated failure")


def _tiers(result):
    return [(a.tier, a.outcome) for a in result.narrative_log]


def test_autogen_wins_when_its_text_is_grounded(stage7):
    result = stage7(autogen=lambda *a, **k: _grounded_text(), llm=_boom)
    assert result.narrative_tier == "autogen"
    assert result.narrative == _grounded_text()
    assert _tiers(result) == [("autogen", "used")]


def test_ungrounded_autogen_falls_to_llm(stage7):
    result = stage7(
        autogen=lambda *a, **k: "Attrition falls by 88.8 points.", llm=lambda *a, **k: _grounded_text()
    )
    assert result.narrative_tier == "llm"
    assert _tiers(result) == [("autogen", "rejected"), ("llm", "used")]
    assert "88.8" in result.narrative_log[0].detail


def test_jargon_in_a_tier_is_rejected_with_the_reason_logged(stage7):
    result = stage7(
        autogen=lambda *a, **k: "The average treatment effect is 2.5.", llm=lambda *a, **k: _grounded_text()
    )
    assert _tiers(result) == [("autogen", "rejected"), ("llm", "used")]
    assert "jargon" in result.narrative_log[0].detail


def test_both_rejected_or_failing_ends_on_template(stage7):
    result = stage7(autogen=lambda *a, **k: "Attrition falls by 88.8 points.", llm=_boom)
    assert result.narrative_tier == "template"
    assert result.narrative.startswith("Top-line drivers of attrition")
    assert _tiers(result) == [("autogen", "rejected"), ("llm", "error"), ("template", "used")]
    assert "simulated failure" in result.narrative_log[1].detail


def test_no_api_key_skips_both_llm_tiers(stage7, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = stage7(autogen=_boom, llm=_boom)
    assert result.narrative_tier == "template"
    assert _tiers(result) == [("autogen", "skipped"), ("llm", "skipped"), ("template", "used")]


def test_missing_autogen_package_is_skipped_not_fatal(stage7, monkeypatch):
    monkeypatch.setattr(narrative, "autogen_available", lambda: False)
    result = stage7(autogen=_boom, llm=lambda *a, **k: _grounded_text())
    assert _tiers(result) == [("autogen", "skipped"), ("llm", "used")]
    assert "not installed" in result.narrative_log[0].detail


def test_tiers_config_selects_and_orders(stage7):
    result = stage7(autogen=_boom, llm=lambda *a, **k: _grounded_text(), config_update={"tiers": ["llm"]})
    assert _tiers(result) == [("llm", "used")]


def test_unknown_tier_raises(stage7):
    with pytest.raises(ValueError, match="unknown explanation tier"):
        stage7(config_update={"tiers": ["autogen", "gpt"]})


def test_required_points_mirror_the_checked_caveats():
    text = narrative.required_points(["agent_internal"], True)
    assert "agent_internal" in text and "no evidence of an effect beyond noise" in text and "fairness" in text
    assert narrative.required_points([], False) == ""


def test_required_points_reach_both_llm_tiers(stage7, monkeypatch):
    monkeypatch.setattr(narrative, "required_points", lambda failed, fairness, observational=False, unpriced_roi=False: "- CAVEAT-X")
    seen = {}

    def fake_autogen(facts, **k):
        seen["autogen"] = facts
        return "Attrition falls by 88.8 points."  # ungrounded, so the chain moves on to tier 2

    def fake_llm(*a, requirements="", **k):
        seen["llm"] = requirements
        return _grounded_text()

    stage7(autogen=fake_autogen, llm=fake_llm)
    assert "CAVEAT-X" in seen["autogen"] and "CAVEAT-X" in seen["llm"]


# --- observational domains: association wording, no causal claims ---

OBS_GOOD = (
    "Pay is associated with lower attrition, by about 2.5 percentage points. Workload shows no "
    "evidence of an effect beyond noise. The $40,000 pay plan has an ROI of 12.5 but raises a "
    "fairness concern (ratio 0.69). These are associations in observational data, not proven causes."
)


def _obs(text):
    return narrative.check_grounding(text, FACTS, observational=True, **CAVEATS)


def test_observational_text_must_say_association():
    report = _obs(GOOD)  # grounded and complete, but never says it is only an association
    assert not report.passed and "association-not-causation" in report.detail()
    assert narrative.check_grounding(GOOD, FACTS, **CAVEATS).passed  # an ordinary domain is unaffected
    assert _obs(OBS_GOOD).passed, _obs(OBS_GOOD).detail()


@pytest.mark.parametrize("phrase", [
    "Pay is a key factor driving attrition down",
    "Higher pay causes lower attrition",
    "Pay is the main driver of attrition",
    "Pay changes lead to lower attrition",
    "Pay is contributing to lower attrition",
])
def test_observational_causal_wording_is_rejected_even_next_to_an_association_caveat(phrase):
    report = _obs(f"{phrase}, about 2.5 percentage points, an association in observational data. "
                  "Workload shows no evidence of an effect beyond noise. The fairness concern is ratio 0.69.")
    assert not report.passed and report.causal_wording and "causal wording" in report.detail()
    # the same sentence is fine for a domain with ground truth or randomization
    plain = narrative.check_grounding(f"{phrase}, about 2.5 percentage points. Workload shows no evidence of an "
                                      "effect beyond noise. The fairness concern is ratio 0.69.", FACTS, **CAVEATS)
    assert plain.passed and not plain.causal_wording


def test_a_negated_cause_is_the_caveat_not_the_overclaim():
    assert _obs(OBS_GOOD).causal_wording == []
    assert _obs("These do not prove that pay causes anything. " + OBS_GOOD).causal_wording == []
    assert _obs("Pay causes lower attrition. " + OBS_GOOD).causal_wording == ["causes"]


def test_required_points_ask_for_association_wording_only_when_observational():
    text = narrative.required_points([], False, observational=True)
    assert "associations in observational data" in text and "'drives'" in text
    assert narrative.required_points([], False) == ""


def test_template_for_an_observational_domain_says_association_and_passes_the_check():
    estimates = [EffectEstimate(treatment="t", outcome="y", ate=0.1, estimator="e", refutation_passed=True)]
    config = {"domain": {"id": "d"}, "ingestion": {"outcome_column": "y"}, "explanation": {"observational": True}}
    text = explanation._template_narrative(config, estimates, [], [], {"a": 1.0})
    assert text.startswith("Associations with") and "not proven causes" in text and "causal effect size" not in text
    assert explanation._grounding(text, text, estimates, [], observational=True).passed
    ordinary = explanation._template_narrative(
        {"domain": {"id": "d"}, "ingestion": {"outcome_column": "y"}}, estimates, [], [], {"a": 1.0})
    assert ordinary.startswith("Top-line drivers of")  # unchanged for domains without the flag


def test_causal_narrative_is_rejected_for_an_observational_domain_and_the_next_tier_wins(stage7):
    causal = "Attrition drivers were ranked by their causal effect."  # grounded, but no association caveat
    result = stage7(
        autogen=lambda *a, **k: causal,
        llm=lambda *a, **k: "Attrition is associated with pay, though these are not proven causes.",
        config_update={"observational": True},
    )
    assert _tiers(result) == [("autogen", "rejected"), ("llm", "used")]
    assert "association-not-causation" in result.narrative_log[0].detail
    # without the flag the same text is accepted by tier 1 (the existing behaviour)
    assert stage7(autogen=lambda *a, **k: causal, llm=_boom).narrative_tier == "autogen"


def test_observational_requirement_reaches_both_llm_tiers(stage7):
    seen = {}

    def fake_autogen(facts, **k):
        seen["autogen"] = facts
        return "Attrition falls by 88.8 points."  # ungrounded, so the chain moves on to tier 2

    def fake_llm(*a, requirements="", **k):
        seen["llm"] = requirements
        return "Attrition is associated with pay, though these are not proven causes."

    stage7(autogen=fake_autogen, llm=fake_llm, config_update={"observational": True})
    assert "associations in observational data" in seen["autogen"] and "associations in observational data" in seen["llm"]


def test_only_the_real_observational_domains_are_flagged(config_loader):
    flagged = {d: bool(config_loader.get_domain(d).extra["explanation"].get("observational"))
               for d in ("freddie_mac", "german_credit", "carclaims", "illinois_wellness", "employee_attrition")}
    assert flagged == {"freddie_mac": True, "german_credit": True, "carclaims": True,
                       "illinois_wellness": False, "employee_attrition": False}


# --- ROI is an outcome drop per dollar, not a return ratio: the narrative must not judge the action either way ---


@pytest.mark.parametrize("phrase", [
    "is favorable", "is a worthwhile investment", "is a good investment", "pays off", "has a strong return",
    "is cost-effective", "looks promising",
])
def test_favorable_wording_is_rejected_because_the_outcome_is_not_priced(phrase):
    text = f"Raising pay lowers attrition by 2.5 percentage points. The plan {phrase} (ROI 12.5)."
    assert narrative.check_grounding(text, FACTS, unpriced_roi=True).roi_wording
    assert not narrative.check_grounding(text, FACTS, unpriced_roi=True).passed
    assert narrative.check_grounding(text, FACTS).passed  # the rule applies only when an action is recommended


@pytest.mark.parametrize("phrase", [
    "has benefits smaller than its cost", "costs more than it returns", "is not worthwhile", "is not worth the money",
    "has an expected benefit less than its cost", "is unfavorable", "is not a practical choice", "has costs that outweigh it",
])
def test_the_opposite_judgment_is_rejected_too(phrase):
    # A live narrative said the benefit was "less than its cost": benefit is a drop in a rate and cost is dollars.
    text = f"Raising pay lowers attrition by 2.5 percentage points. The plan {phrase}."
    report = narrative.check_grounding(text, FACTS, unpriced_roi=True)
    assert report.roi_wording and not report.passed
    assert "outcome is not priced" in report.detail()


@pytest.mark.parametrize("text", [
    "Raising pay is expected to lower attrition by 2.5 percentage points at a cost of $40,000.",
    "The pay plan ranks first of the three actions by reduction per dollar.",
])
def test_plain_statements_of_effect_and_cost_pass(text):
    assert not narrative.check_grounding(text, FACTS, unpriced_roi=True).roi_wording


def test_required_points_explain_the_unpriced_roi_only_when_an_action_is_recommended():
    points = narrative.required_points([], False, unpriced_roi=True)
    assert "no price" in points and "smaller or larger than its cost" in points
    assert narrative.required_points([], False) == ""


def test_the_unpriced_roi_rule_applies_exactly_when_an_action_is_recommended():
    rec = lambda recommended: [SimpleNamespace(roi=0.0000391, recommended=recommended)]
    assert explanation._has_recommended_action(rec(True))
    assert not explanation._has_recommended_action(rec(False))
    assert not explanation._has_recommended_action([])


def test_the_fact_sheet_states_the_effect_and_cost_in_real_units_and_that_roi_is_unpriced(domain_config):
    from causal_engine.models.schemas import InterventionRecommendation

    rec = InterventionRecommendation(
        id="manager_training", target_variable="manager_quality", expected_effect=0.0426, cost=15000.0,
        roi=0.0426 / 15000, fairness_ratio=0.99, rank=1,
    )

    facts = explanation._template_narrative(domain_config, [], [], [rec], {"manager_quality": 1.0})

    assert "expected to lower attrition by 4.26 percentage points" in facts
    assert "cost=$15,000" in facts
    assert "has no dollar value here" in facts

