"""Does the PDF / image ingestion path work, on the three real domains' synthetic attachments?

Two questions, both about the PLUMBING (the attachments are synthetic and carry no new
information: see attachment_synth.py):

  1. Extraction validity. The attachments were rendered from tabular columns, so those columns
     are the ground truth for what extraction should recover:
       PDF    does a keyword group's count agree with the field the sentence was rendered from?
       image  can a linear model predict the rendered quantity (damage level, gym visits, loan
              term, ...) from the extracted features, in cross-validation? Reported for the
              OpenCV statistics alone, the CNN embedding alone and both, against a shuffled-target
              control that should sit at or below zero.
  2. Pipeline. The same rows run through Stages 1-6 twice: with the attachment features added (and
     adjusted for in Stage 4), and without. Because the features hold nothing new, the estimates
     should barely move; a large move would mean the plumbing changed something it should not.

It says nothing about real claim photos, reports or scans.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rootcause.evaluation import attachment_synth
from rootcause.pipeline import modalities, runner
from rootcause.utils.config_loader import ConfigLoader

DOMAINS = ("carclaims", "illinois_wellness", "german_credit")
TAG_MULTIMODAL = "eval_multimodal"
TAG_CONTROL = "eval_multimodal_control"  # not an attachment-bearing kind: the same rows, no attachments
CV_FOLDS = 5


# (feature column, boolean truth from the tabular row, description)
PDF_CHECKS: dict[str, list[tuple[str, Callable[[pd.DataFrame], pd.Series], str]]] = {
    "carclaims": [
        ("claim_report__kw_fault", lambda d: d["fault_policy_holder"] == 1, "policy holder at fault"),
        ("claim_report__kw_rural", lambda d: d["accident_rural"] == 1, "rural accident area"),
        ("claim_report__kw_witness", lambda d: d["witness_present"] == 1, "witness present"),
    ],
    "illinois_wellness": [
        ("screening_summary__kw_male", lambda d: d["male"] == 1, "male"),
        ("screening_summary__kw_age50", lambda d: d["age50"] == 1, "age 50+"),
        ("screening_summary__kw_age37_49", lambda d: d["age37_49"] == 1, "age 37-49"),
    ],
    "german_credit": [
        ("loan_summary__kw_business", lambda d: d["purpose"] == "business", "purpose business"),
        ("loan_summary__kw_education", lambda d: d["purpose"] == "education", "purpose education"),
        ("loan_summary__kw_car", lambda d: d["purpose"].isin(["car_new", "car_used"]), "purpose car"),
        ("loan_summary__kw_own_home", lambda d: d["housing"] == "own", "owns home"),
    ],
}

# (attachment name, what was drawn, the tabular quantity the image encodes)
IMAGE_PROBES: dict[str, list[tuple[str, str, Callable[[pd.DataFrame], pd.Series]]]] = {
    "carclaims": [
        ("damage_photo", "damage level (blotches)", lambda d: d["n_supplements"].map(attachment_synth.DAMAGE_LEVELS).astype(float)),
    ],
    "illinois_wellness": [
        ("activity_chart", "gym visits (bar height)", lambda d: d["gym_0815_0716"].clip(upper=120).astype(float)),
        ("activity_chart", "sick leave days (bar height)", lambda d: d["sickleave_0815_0716"].clip(upper=40).astype(float)),
    ],
    "german_credit": [
        ("profile_chart", "applicant age (bar height)", lambda d: d["age"].clip(upper=80).astype(float)),
        ("profile_chart", "years employed, 0-4 (bar height)", lambda d: d["employment_since"].map(attachment_synth.EMPLOYMENT_LEVELS).astype(float)),
    ],
}


@dataclass
class PdfRow:
    domain: str
    feature: str
    description: str
    n: int
    agreement: float  # share of records where (keyword count > 0) equals the tabular truth


@dataclass
class ImageRow:
    domain: str
    attachment: str
    target: str
    n: int
    r2_stats: float
    r2_cnn: float
    r2_both: float
    r2_shuffled: float  # both feature sets, target shuffled: should be near or below zero


@dataclass
class EstimateRow:
    domain: str
    treatment: str
    ate_control: float
    ate_multimodal: float
    placebo_control: bool | None
    placebo_multimodal: bool | None


@dataclass
class DomainResult:
    domain: str
    n_rows: int
    n_attachment_features: int
    seconds: float
    pdf: list[PdfRow] = field(default_factory=list)
    images: list[ImageRow] = field(default_factory=list)
    estimates: list[EstimateRow] = field(default_factory=list)
    fairness_control: float | None = None
    fairness_multimodal: float | None = None
    top_control: str | None = None
    top_multimodal: str | None = None
    missing_files: int = 0


def cv_r2(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    """Cross-validated R^2 of a ridge regression (features standardized)."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 12)))
    pred = cross_val_predict(model, X, y, cv=CV_FOLDS)
    return float(1.0 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))


def image_probe(df: pd.DataFrame, attachment: str, truth: pd.Series, seed: int = 0) -> tuple[float, float, float, float]:
    columns = [c for c in df.columns if c.startswith(f"{attachment}__")]
    stats = [c for c in columns if c.split("__")[1] in modalities.IMAGE_STAT_FEATURES]
    cnn = [c for c in columns if c.split("__")[1].startswith("cnn")]
    keep = df[columns].notna().all(axis=1) & truth.notna()
    y = truth[keep].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    return (
        cv_r2(df.loc[keep, stats].to_numpy(dtype=float), y) if stats else float("nan"),
        cv_r2(df.loc[keep, cnn].to_numpy(dtype=float), y) if cnn else float("nan"),
        cv_r2(df.loc[keep, columns].to_numpy(dtype=float), y),
        cv_r2(df.loc[keep, columns].to_numpy(dtype=float), rng.permutation(y)),
    )


def evaluate_domain(domain_id: str, loader: ConfigLoader | None = None) -> DomainResult:
    start = time.perf_counter()
    domain = (loader or ConfigLoader()).get_domain(domain_id)
    cfg_mm = runner.with_dataset(domain.extra, TAG_MULTIMODAL)
    cfg_ctl = runner.with_dataset(domain.extra, TAG_CONTROL)
    path = runner.resolve_data_path(domain.extra, "multimodal")
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run scripts/generate_attachments.py first")

    with_att = runner.run_analysis_stages(cfg_mm, path)
    control = runner.run_analysis_stages(cfg_ctl, path)
    raw = with_att.raw
    added = modalities.attachment_columns(cfg_mm)
    result = DomainResult(
        domain=domain_id,
        n_rows=len(raw),
        n_attachment_features=len(added),
        seconds=0.0,
        missing_files=int(raw[added].isna().all(axis=1).sum()),
    )

    for feature, truth_fn, description in PDF_CHECKS[domain_id]:
        truth = truth_fn(raw)
        result.pdf.append(PdfRow(domain_id, feature, description, len(raw), float(((raw[feature] > 0) == truth).mean())))
    for attachment, target, truth_fn in IMAGE_PROBES[domain_id]:
        r2 = image_probe(raw, attachment, truth_fn(raw))
        result.images.append(ImageRow(domain_id, attachment, target, len(raw), *r2))

    by_treatment = {e.treatment: e for e in control.effects}
    for est in with_att.effects:
        base = by_treatment[est.treatment]
        result.estimates.append(
            EstimateRow(domain_id, est.treatment, base.ate, est.ate, base.refutation_passed, est.refutation_passed)
        )
    if with_att.recommendations and control.recommendations:
        result.fairness_multimodal = with_att.recommendations[0].fairness_ratio
        result.fairness_control = control.recommendations[0].fairness_ratio
        result.top_multimodal = with_att.recommendations[0].id
        result.top_control = control.recommendations[0].id
    result.seconds = time.perf_counter() - start
    return result


def run(domains=DOMAINS) -> list[DomainResult]:
    loader = ConfigLoader()
    return [evaluate_domain(d, loader) for d in domains]


# --- report -------------------------------------------------------------------------------


def render_markdown(results: list[DomainResult]) -> str:
    lines = [
        "# Multimodal ingestion check (synthetic PDF and image attachments)",
        "",
        "Generated by `python -m rootcause.evaluation --multimodal`. Every attachment is **synthetic**: rendered from its own "
        "record's pre-outcome tabular covariates by `scripts/generate_attachments.py`, never from the outcome or the randomized "
        "treatment. They hold no information the tabular columns do not, so this shows that the PDF / image path works, not "
        "that documents or photos help a causal question, and it says nothing about real claim photos or reports.",
        "",
        "| Domain | Records | Attachment features added | Records with no attachment | Time |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(f"| {r.domain} | {r.n_rows:,} | {r.n_attachment_features} | {r.missing_files} | {r.seconds:.0f}s |")
    lines += [
        "",
        "## PDF text extraction (pypdf)",
        "",
        "Share of records where the keyword group's count is positive exactly when the tabular field says it should be (the sentence was rendered from that field).",
        "",
        "| Domain | Feature | Field | Records | Agreement |",
        "|---|---|---|---|---|",
    ]
    for r in results:
        for row in r.pdf:
            lines.append(f"| {r.domain} | `{row.feature}` | {row.description} | {row.n:,} | {row.agreement:.1%} |")
    lines += [
        "",
        "## Image features (Pillow + OpenCV + ResNet18)",
        "",
        "Cross-validated R² of a ridge regression predicting the quantity the image was drawn from. **Stats** = brightness, contrast, "
        "edge density, sharpness; **CNN** = 8 PCA components of the pretrained ResNet18 embedding; **Shuffled** = both, target permuted "
        "(should be near 0 or below).",
        "",
        "| Domain | Image | Drawn from | Records | R² stats | R² CNN | R² both | R² shuffled |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        for row in r.images:
            lines.append(
                f"| {r.domain} | {row.attachment} | {row.target} | {row.n:,} | {row.r2_stats:.2f} | {row.r2_cnn:.2f} | {row.r2_both:.2f} | {row.r2_shuffled:.2f} |"
            )
    lines += [
        "",
        "## Stages 1-6 with and without the attachment features",
        "",
        "Same rows, same config; the second run adds the PDF/image feature columns (`adjust_for_attachments` also puts them in Stage 4's adjustment set). "
        "Because the columns repeat information already in the table, the estimates should barely move.",
        "",
        "| Domain | Treatment | ATE without | ATE with | Change | Placebo without / with |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        for e in r.estimates:
            passed = lambda v: "–" if v is None else ("pass" if v else "fail")  # noqa: E731
            lines.append(
                f"| {r.domain} | {e.treatment} | {e.ate_control:+.4f} | {e.ate_multimodal:+.4f} | {e.ate_multimodal - e.ate_control:+.4f} | {passed(e.placebo_control)} / {passed(e.placebo_multimodal)} |"
            )
    lines += ["", "| Domain | Fairness ratio without / with | Top-ranked intervention without / with |", "|---|---|---|"]
    for r in results:
        if r.fairness_control is not None:
            lines.append(f"| {r.domain} | {r.fairness_control:.3f} / {r.fairness_multimodal:.3f} | {r.top_control} / {r.top_multimodal} |")
    lines += [
        "",
        "## How to read this",
        "",
        "- PDF agreement near 100% means the text extraction and keyword counting work; it is high by construction because the templates are regular. Real documents are messier, and keyword groups would need to be written for them.",
        "- Image R² measures whether the extracted numbers carry what was drawn. The OpenCV statistics capture coarse properties (how much ink, how many edges); the ImageNet embedding is generic, so it can miss a domain-specific cue that a hand-built statistic sees, or the reverse. Both are reported so neither is assumed.",
        "- The pipeline comparison is a regression check on the plumbing. The changes in the estimates are noise from adding redundant columns, not a finding about the domains.",
        "- Embeddings are reduced by PCA fitted on the ingested rows (unsupervised); columns of one run are not comparable with another's.",
    ]
    return "\n".join(lines) + "\n"


def render_json(results: list[DomainResult]) -> str:
    return json.dumps([asdict(r) for r in results], indent=2)
