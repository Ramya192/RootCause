"""Stage 7 narrative tiers (rootcause/pipeline/narrative.py + explanation.py).

No test touches the network: the AutoGen chain runs against AutoGen's replay client and
the single-call tier against a stub, so the suite stays free and hermetic.
"""

from __future__ import annotations

import asyncio

import pytest

from rootcause.models.schemas import EffectEstimate, InterventionRecommendation
from rootcause.pipeline import counterfactuals, explanation, interventions, narrative

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
    monkeypatch.setattr(narrative, "required_points", lambda failed, fairness: "- CAVEAT-X")
    seen = {}

    def fake_autogen(facts, **k):
        seen["autogen"] = facts
        return "Attrition falls by 88.8 points."  # ungrounded, so the chain moves on to tier 2

    def fake_llm(*a, requirements="", **k):
        seen["llm"] = requirements
        return _grounded_text()

    stage7(autogen=fake_autogen, llm=fake_llm)
    assert "CAVEAT-X" in seen["autogen"] and "CAVEAT-X" in seen["llm"]
