"""Stage 7 helpers: grounding checks and the AutoGen narrative team.

Stage 7 tries three narrative writers in order (see explanation.py) and keeps the
first whose text passes `check_grounding`:

  1. `autogen`: a three-agent chain (analyst drafts, skeptic audits the draft against
     the facts, writer finalizes), run for a fixed number of turns.
  2. `llm`: one OpenAI call over the same facts.
  3. `template`: deterministic text built from the facts; needs no API key.

Choosing between them is deliberately NOT done by asking an LLM which text is better
(unreliable, and another paid call). The check is programmatic instead:

  * every number in the text must appear in the fact sheet (as-is, or x100 as a
    percentage, to the precision the text shows), and
  * for the LLM tiers, the text must be plain language (no "ATE", "SHAP", "p-value",
    "placebo", or "statistically significant", which the facts use but the audience should
    not have to decode), and
  * the required caveats must be present: a failed placebo check must be described as
    "no evidence of an effect beyond noise", and a fairness flag must be mentioned, and
  * for a domain flagged `explanation.observational` (real data with no randomization or
    ground truth), the text must say the findings are associations and must not use
    causal wording ("drives", "causes", "leads to", "key factor", "contributing to"), and
  * when the recommended action's ROI is below 1 (benefit smaller than cost), the text must not
    call it favorable ("favorable", "worthwhile", "good investment", "pays off", ...).

What the check does NOT catch: a wrong claim that uses only correct numbers (for example
attributing an effect to the wrong lever; the ROI-below-1 rule above covers one known case), or an overclaim phrased without numbers or a
hedge word. It is a guard against invented figures and dropped caveats, not a proof that
the narrative is right.

AutoGen is an optional dependency: if it is not installed, tier 1 is skipped and the
log says so.
"""

from __future__ import annotations

import asyncio
import re
from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

# Numbers in prose: 1,200 / 0.0248 / -3.5 / $700 / 12.5%. A digit run glued to letters or
# underscores (a column name such as deductible_100usd or terminated_0119) is not a number.
_NUMBER = re.compile(r"(?<![\w.])-?\$?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?%?(?!\w)")
# Digit runs anywhere in the facts, including inside column names (age37_49): a narrative that
# says "ages 37 to 49" is quoting the facts, not inventing numbers.
_DIGIT_RUN = re.compile(r"\d+(?:\.\d+)?")
# "not statistically significant" is the plain-language way to say a check failed, not jargon.
_NEGATED_SIGNIFICANCE = re.compile(r"\b(?:not|no|never)\s+(?:statistically\s+)?significan\w*", re.IGNORECASE)

# Words that acknowledge "no evidence of an effect" / "could be noise".
_NO_EFFECT_HEDGE = re.compile(
    r"no (?:clear |reliable |detectable |real |strong |statistical )?evidence"
    r"|not (?:statistically )?(?:distinguishable|significant|reliable|different)"
    r"|indistinguishable|cannot (?:be )?(?:distinguish|rule out|tell)|can(?:'|no)t tell"
    r"|(?:could|may|might) (?:just |simply )?be (?:noise|chance|random)"
    r"|(?:within|beyond) (?:the )?(?:noise|chance)|no (?:measurable|detectable) (?:effect|impact)"
    r"|noise|chance|unreliable|inconclusive",
    re.IGNORECASE,
)
_FAIRNESS_MENTION = re.compile(
    r"fair|disparit|bias|discriminat|equit|protected|disproportion|unequal", re.IGNORECASE
)

# Terms the audience should not have to decode (the fact sheet itself uses them).
_JARGON = re.compile(
    r"\bATE\b|\bCATE\b|\bSHAP\b|average treatment effect|p-value|\bplacebo\b"
    r"|\bpermutation\b|\bstatistical(?:ly)? significan\w*|\bbackdoor\b",
    re.IGNORECASE,
)

# Observational domains: the text must say "association" (or equivalent) and must not credit a lever
# with causing the outcome. A negated mention ("not proven causes") is the caveat, not the overclaim.
_OBSERVATIONAL_MARKER = re.compile(
    r"associat|observational|correlat|confound|unmeasured|not proven|no proof|cannot prove|does not prove"
    r"|not necessarily caus|not (?:a )?(?:proven )?cause",
    re.IGNORECASE,
)
_CAUSAL_NEGATION = re.compile(
    r"\b(?:not|never|nor|isn't|aren't|doesn't|does not|do not|don't|cannot|can't)\b[^.]{0,40}?\bcaus\w+",
    re.IGNORECASE,
)
_CAUSAL_WORDING = re.compile(
    r"\bkey (?:factors?|drivers?)\b|\b(?:main|primary|major|biggest|top) (?:factors?|drivers?|causes?)\b"
    r"|\bdriv(?:e|es|ing|en)\b|\bcaus(?:e|es|ed|ing)\b|\bleads? to\b|\bcontribut\w+ to\b|\bresults? in\b",
    re.IGNORECASE,
)

# ROI = benefit / cost (interventions.py), so ROI < 1 means the action costs more than it returns. Live runs
# called an ROI of 0.0000391 "favorable": correct number, wrong meaning. Negated or comparative uses
# ("not worthwhile", "less favorable") are the honest reading, so they are removed before matching.
_POSITIVE_ROI = re.compile(
    r"\bfavou?rabl[ey]\b|\bworth(?:while| it| the (?:cost|investment|money))\b|\bgood (?:investment|return|value)\b"
    r"|\b(?:strong|solid|healthy|attractive|positive|high|great|excellent) (?:return|roi)\b|\bpays? (?:off|for itself)\b"
    r"|\bcost-effective\b|\bprofitabl[ey]\b|\bbeneficial\b|\bpromising\b|\bsensible\b",
    re.IGNORECASE,
)
_NEGATED_POSITIVE_ROI = re.compile(
    r"\b(?:not|never|no|isn't|aren't|doesn't|does not|hardly|barely|less)\s+(?:\w+\s+){0,2}?"
    r"(?:favou?rabl|worth|good|strong|solid|healthy|attractive|positive|high|great|excellent|pays?|cost-effective|profitabl|beneficial|promising|sensible)\w*",
    re.IGNORECASE,
)

NARRATIVE_TIERS = ("autogen", "llm", "template")


@dataclass
class GroundingReport:
    passed: bool
    ungrounded_numbers: list[str] = field(default_factory=list)
    missing_caveats: list[str] = field(default_factory=list)
    jargon: list[str] = field(default_factory=list)
    causal_wording: list[str] = field(default_factory=list)
    roi_wording: list[str] = field(default_factory=list)

    def detail(self) -> str:
        parts = []
        if self.ungrounded_numbers:
            parts.append("numbers not in the fact sheet: " + ", ".join(self.ungrounded_numbers))
        if self.missing_caveats:
            parts.append("missing caveats: " + ", ".join(self.missing_caveats))
        if self.jargon:
            parts.append("jargon: " + ", ".join(self.jargon))
        if self.causal_wording:
            parts.append("causal wording on observational data: " + ", ".join(self.causal_wording))
        if self.roi_wording:
            parts.append("favorable wording although ROI is below 1 (benefit under cost): " + ", ".join(self.roi_wording))
        return "; ".join(parts) or "ok"


def extract_numbers(text: str) -> list[tuple[str, float, int]]:
    """(raw token, value, decimals shown) for every number in `text`. A trailing '%'
    is dropped and the value kept as printed (so '12%' is 12.0)."""
    found = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        token = raw.replace("$", "").replace(",", "").rstrip("%")
        if token in ("", "-"):
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        decimals = max(0, -Decimal(token).as_tuple().exponent)  # 1.33e-05 shows 7 places
        found.append((raw, value, decimals))
    return found


def _is_grounded(value: float, decimals: int, fact_values: list[float]) -> bool:
    """`value` (shown to `decimals` places) matches a fact, up to rounding, as-is or as a
    percentage. Sign is ignored: prose says "cut by 2.5 points", not "-2.5"."""
    half_unit = 0.5 * 10 ** (-decimals) + 1e-9
    return any(
        abs(abs(value) - abs(f) * scale) <= half_unit
        for f in fact_values
        for scale in (1.0, 100.0)
    )


def check_grounding(
    text: str,
    facts_text: str,
    *,
    failed_refutation: bool = False,
    fairness_flagged: bool = False,
    plain_language: bool = False,
    observational: bool = False,
    roi_below_cost: bool = False,
) -> GroundingReport:
    """Does `text` stay inside `facts_text`? See the module docstring for what it covers."""
    fact_values = [v for _, v, _ in extract_numbers(facts_text)]
    fact_values += [float(d) for d in _DIGIT_RUN.findall(facts_text)]
    ungrounded = [
        raw
        for raw, value, decimals in extract_numbers(text)
        if not _is_grounded(value, decimals, fact_values)
    ]
    missing = []
    if failed_refutation and not _NO_EFFECT_HEDGE.search(text):
        missing.append("no-evidence-of-effect (a placebo check failed)")
    if fairness_flagged and not _FAIRNESS_MENTION.search(text):
        missing.append("fairness flag")
    if not text.strip():
        missing.append("non-empty narrative")
    causal = []
    if observational:
        if text.strip() and not _OBSERVATIONAL_MARKER.search(text):
            missing.append("association-not-causation (observational data)")
        causal = sorted({m.group(0).lower() for m in _CAUSAL_WORDING.finditer(_CAUSAL_NEGATION.sub("", text))})
    roi = []
    if roi_below_cost:
        roi = sorted({m.group(0).lower() for m in _POSITIVE_ROI.finditer(_NEGATED_POSITIVE_ROI.sub("", text))})
    jargon = (
        sorted({m.group(0) for m in _JARGON.finditer(_NEGATED_SIGNIFICANCE.sub('', text))})
        if plain_language
        else []
    )
    return GroundingReport(
        passed=not ungrounded and not missing and not jargon and not causal and not roi,
        ungrounded_numbers=ungrounded,
        missing_caveats=missing,
        jargon=jargon,
        causal_wording=causal,
        roi_wording=roi,
    )


# --- Tier 1: AutoGen analyst -> skeptic -> writer ---

_ANALYST_PROMPT = (
    "You are a causal-inference analyst. From the FACTS in the task, draft a short "
    "(3-5 sentence) plain-language narrative for {audience} about {outcome} among "
    "{entities}. Use ONLY the facts and only numbers that appear in them. Never write "
    "'ATE', 'SHAP', 'placebo', 'p-value' or 'statistically significant': say what a lever changes in "
    "the outcome (e.g. 'about 2.5 percentage points fewer') in business language."
)
_SKEPTIC_PROMPT = (
    "You are a skeptical reviewer. Check the analyst's draft claim by claim against the "
    "FACTS. List every problem: a number that is not in the facts; an association "
    "described as a proven cause; a treatment whose placebo check FAILED described as if "
    "it had an effect (it must say there is no evidence of an effect beyond noise); a "
    "fairness flag that is left out; jargon ('ATE', 'average treatment effect', 'SHAP', "
    "'placebo', 'p-value') or 'statistically significant'. If there are no problems, reply "
    "'No issues'. Be brief."
)
_WRITER_PROMPT = (
    "You are the final writer. Rewrite the analyst's draft applying the reviewer's "
    "corrections. Use ONLY the facts and numbers that appear in them, keep it to 3-5 "
    "sentences of plain language for {audience} (never 'ATE', 'SHAP', 'placebo', 'p-value' "
    "or 'statistically significant'), and keep every required caveat. Reply "
    "with the narrative text only: no headings, no preamble, no mention of the review."
)

# Task message + analyst + skeptic + writer.
_CHAIN_MESSAGES = 4


def _make_autogen_client(model: str, cfg: dict):
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    return OpenAIChatCompletionClient(
        model=model,
        max_tokens=int(cfg.get("max_tokens", 600)),
        temperature=cfg.get("temperature", 0),
        timeout=float(cfg.get("timeout_seconds", 60)),
    )


async def _run_chain(facts_text: str, *, model_client: Any, audience: str, outcome: str, entities: str) -> str:
    from autogen_agentchat.agents import AssistantAgent
    from autogen_agentchat.conditions import MaxMessageTermination
    from autogen_agentchat.teams import RoundRobinGroupChat

    def agent(name: str, prompt: str) -> AssistantAgent:
        return AssistantAgent(
            name,
            model_client=model_client,
            system_message=prompt.format(audience=audience, outcome=outcome, entities=entities),
        )

    team = RoundRobinGroupChat(
        [
            agent("analyst", _ANALYST_PROMPT),
            agent("skeptic", _SKEPTIC_PROMPT),
            agent("writer", _WRITER_PROMPT),
        ],
        termination_condition=MaxMessageTermination(_CHAIN_MESSAGES),
    )
    result = await team.run(task=f"FACTS:\n{facts_text}\n\nAudience: {audience}.")
    final = [
        m for m in result.messages
        if getattr(m, "source", None) == "writer" and isinstance(getattr(m, "content", None), str)
    ]
    if not final or not final[-1].content.strip():
        raise RuntimeError("AutoGen chain ended without a writer message")
    return final[-1].content.strip()


def _run_sync(coro):
    """Run a coroutine to completion from sync code, even if a loop is already running
    in this thread (the API worker thread has none; a notebook or pytest-asyncio does)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def required_points(
    failed_treatments: list[str],
    fairness_flagged: bool,
    observational: bool = False,
    roi_below_cost: bool = False,
) -> str:
    """The caveats `check_grounding` will demand, phrased as an instruction, so a writer is
    told what it will be checked on instead of finding out by being rejected."""
    points = []
    if failed_treatments:
        points.append(
            "State plainly that there is no evidence of an effect beyond noise for: "
            + ", ".join(failed_treatments)
            + " (its check failed); do not present it as a lever that works."
        )
    if fairness_flagged:
        points.append("Mention the fairness / disparity concern for the top recommendation.")
    if observational:
        points.append(
            "Say these are associations in observational data, not proven causes: write 'associated with', never "
            "say a lever 'drives', 'causes', 'leads to' or is a 'key factor' in the outcome, and say the true "
            "effect could differ because other factors were not measured."
        )
    if roi_below_cost:
        points.append(
            "The recommended action's ROI is below 1, meaning its estimated benefit is smaller than its cost: say so "
            "plainly and do not call it favorable, worthwhile, attractive or a good return."
        )
    return "\n".join(f"- {p}" for p in points)


def autogen_available() -> bool:
    try:
        import autogen_agentchat  # noqa: F401
        import autogen_ext.models.openai  # noqa: F401
    except ImportError:
        return False
    return True


def autogen_narrative(
    facts_text: str,
    *,
    model: str,
    audience: str,
    outcome: str,
    entities: str,
    cfg: Optional[dict] = None,
    model_client: Any = None,
) -> str:
    """Analyst -> skeptic -> writer over `facts_text`; returns the writer's text.

    Exactly three model turns (a hard cap, not a convergence loop), each limited to
    `max_tokens`. `model_client` is injectable so tests can replay canned replies.
    """
    cfg = cfg or {}
    client = model_client if model_client is not None else _make_autogen_client(model, cfg)

    async def _go() -> str:
        try:
            return await asyncio.wait_for(
                _run_chain(
                    facts_text, model_client=client, audience=audience,
                    outcome=outcome, entities=entities,
                ),
                timeout=float(cfg.get("total_timeout_seconds", 180)),
            )
        finally:
            close = getattr(client, "close", None)
            if close is not None:
                await close()

    return _run_sync(_go())
