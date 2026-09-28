"""Synthetic PDF and image attachments for the three real domains.

None of the real datasets ships documents or photos. To exercise the multimodal ingestion path
on realistic-looking inputs, each record gets a document and an image RENDERED from its own
tabular covariates:

  carclaims          claim_report PDF (fault, accident area, witness, vehicle, supplements) and
                     a damage_photo whose amount of damage follows the number of supplements
  illinois_wellness  screening_summary PDF (sex, age band, baseline sick leave and gym visits)
                     and an activity_chart image whose bars follow sick leave and gym visits
  german_credit      loan_summary PDF (purpose, housing, employment, savings, age) and a
                     profile_chart image whose bars follow age and length of employment

What this is and is not. The attachments carry NO information the tabular columns do not
already hold, and are drawn only from covariates that are neither the outcome (`fraud_found`,
`default`, `terminated_0119`) nor a treatment: not the randomized `treat`, and not the levers Stage 4
estimates (police report, agent, deductible; loan term, amount, installment rate). Rendering the
outcome would leak the answer; rendering a lever would put a copy of the treatment among the
features, and adjusting for a copy of the treatment removes the effect (a "bad control"). So they cannot make an estimate better. What they can show is
that the ingestion path works end to end: a PDF's text and an image's pixels come back as
numbers that recover what was rendered (see evaluation/multimodal_eval.py), and those numbers
flow through Stages 2-6 without breaking anything. Nothing here says anything about real
claim photos or real reports.

Rendering is deterministic: each record's randomness is seeded by (seed, record id).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

SEED = 2026
IMAGE_SIZE = 224

FILLER = (
    "The file was opened on the next business day.",
    "All supporting documents were received by mail.",
    "The reviewer noted no unusual circumstances in the submission.",
    "A follow-up call was scheduled with the customer.",
    "Contact details on file were confirmed.",
    "This summary was generated automatically from the case record.",
    "Standard processing time applies to this request.",
)


def _rng(seed: int, record_id) -> np.random.Generator:
    return np.random.default_rng([seed, int(record_id)])


def _filler(rng: np.random.Generator, k: int = 3) -> list[str]:
    return [FILLER[i] for i in rng.choice(len(FILLER), size=k, replace=False)]


# --- documents (lists of lines) -----------------------------------------------------------


def carclaims_report(row: pd.Series, rng: np.random.Generator) -> list[str]:
    return [
        f"CLAIM REPORT {int(row['claim_id'])}",
        f"Policyholder age group: {row['policyholder_age_group'].replace('_', ' ')}.",
        f"Vehicle: {row['vehicle_category']}, price band {row['vehicle_price'].replace('_', ' ')}, age {row['vehicle_age']}.",
        f"Accident area: {'rural' if row['accident_rural'] else 'urban'}.",
        f"Fault: {'policy holder' if row['fault_policy_holder'] else 'third party'}.",
        "A witness was present." if row["witness_present"] else "No witness was present.",
        f"Supplements filed: {row['n_supplements'].replace('_', ' ')}.",
        *_filler(rng),
    ]


def wellness_summary(row: pd.Series, rng: np.random.Generator) -> list[str]:
    age = "50 plus" if row["age50"] else ("37 to 49" if row["age37_49"] else "under 37")
    return [
        f"SCREENING SUMMARY FOR PARTICIPANT {int(row['employee_id'])}",
        f"Sex: {'male' if row['male'] else 'female'}.",
        f"Age band: {age}.",
        f"Baseline sick leave: {row['sickleave_0815_0716']:.1f} days.",
        f"Baseline gym visits: {int(row['gym_0815_0716'])}.",
        *_filler(rng),
    ]


def loan_summary(row: pd.Series, rng: np.random.Generator) -> list[str]:
    return [
        f"LOAN APPLICATION SUMMARY {int(row['applicant_id'])}",
        f"Purpose: {row['purpose'].replace('_', ' ')}.",
        f"Housing: {row['housing']}.",
        f"Employed for: {row['employment_since'].replace('_', ' ')}.",
        f"Savings: {row['savings'].replace('_', ' ')}.",
        f"Applicant age: {int(row['age'])}.",
        *_filler(rng),
    ]


# --- images -------------------------------------------------------------------------------


def _canvas(rng: np.random.Generator, base: int) -> "Image.Image":
    from PIL import Image

    tone = np.clip(base + rng.integers(-15, 16), 0, 255)
    return Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (tone, tone, min(255, tone + 8)))


def _noisy(img, rng: np.random.Generator, sd: float = 6.0) -> np.ndarray:
    arr = np.asarray(img).astype(float) + rng.normal(0.0, sd, (IMAGE_SIZE, IMAGE_SIZE, 3))
    return np.clip(arr, 0, 255).astype(np.uint8)


DAMAGE_LEVELS = {"none": 0, "1_to_2": 1, "3_to_5": 2, "gt_5": 3}


def damage_photo(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """A car on a road; the number of dark blotches on the body follows the supplement level."""
    from PIL import ImageDraw

    img = _canvas(rng, 190)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 150, IMAGE_SIZE, IMAGE_SIZE], fill=(90, 90, 95))  # road
    body = tuple(int(c) for c in rng.integers(60, 220, 3))
    draw.rectangle([30, 90, 194, 150], fill=body, outline=(20, 20, 20), width=2)
    draw.polygon([(60, 90), (90, 55), (150, 55), (174, 90)], fill=body, outline=(20, 20, 20))
    for cx in (65, 160):
        draw.ellipse([cx - 16, 135, cx + 16, 167], fill=(25, 25, 25))
    level = DAMAGE_LEVELS[row["n_supplements"]]
    for _ in range(level * 4):
        x, y, r = rng.integers(35, 190), rng.integers(95, 145), rng.integers(4, 11)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(30, 10, 10))
    return _noisy(img, rng)


def activity_chart(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Two bars: baseline sick leave (of 40 days) and baseline gym visits (of 120)."""
    from PIL import ImageDraw

    img = _canvas(rng, 235)
    draw = ImageDraw.Draw(img)
    draw.line([(20, 200), (204, 200)], fill=(0, 0, 0), width=2)
    draw.line([(20, 200), (20, 20)], fill=(0, 0, 0), width=2)
    sick = min(row["sickleave_0815_0716"] / 40.0, 1.0)
    gym = min(row["gym_0815_0716"] / 120.0, 1.0)
    draw.rectangle([50, 200 - int(170 * sick), 100, 200], fill=(200, 60, 60))
    draw.rectangle([124, 200 - int(170 * gym), 174, 200], fill=(60, 60, 200))
    return _noisy(img, rng, 4.0)


EMPLOYMENT_LEVELS = {"unemployed": 0, "lt_1y": 1, "1_to_4y": 2, "4_to_7y": 3, "ge_7y": 4}


def profile_chart(row: pd.Series, rng: np.random.Generator) -> np.ndarray:
    """Two bars: the applicant's age (of 80) and length of employment (of 4 steps)."""
    from PIL import ImageDraw

    img = _canvas(rng, 240)
    draw = ImageDraw.Draw(img)
    draw.line([(20, 205), (204, 205)], fill=(0, 0, 0), width=2)
    draw.line([(20, 205), (20, 20)], fill=(0, 0, 0), width=2)
    age = min(row["age"] / 80.0, 1.0)
    employment = EMPLOYMENT_LEVELS[row["employment_since"]] / 4.0
    draw.rectangle([50, 205 - int(180 * age), 100, 205], fill=(40, 120, 60))
    draw.rectangle([124, 205 - int(180 * employment), 174, 205], fill=(150, 90, 30))
    return _noisy(img, rng, 4.0)


# --- per-domain registry ------------------------------------------------------------------


@dataclass(frozen=True)
class DomainAttachments:
    domain_id: str
    id_column: str
    source: str  # repo-relative CSV the rows come from
    sample_output: str | None  # where a row sample is written (None: use every row, no new CSV)
    sample_size: int | None
    pdf_dir: str
    image_dir: str
    document: Callable[[pd.Series, np.random.Generator], list[str]]
    image: Callable[[pd.Series, np.random.Generator], np.ndarray]
    # tabular columns the attachments are rendered from (never the outcome or a randomized treatment)
    sources: tuple[str, ...]
    forbidden: tuple[str, ...]  # columns that must NOT influence the rendering


DOMAINS = {
    "carclaims": DomainAttachments(
        "carclaims", "claim_id", "data/carclaims/claims.csv", "data/carclaims/claims_multimodal.csv", 5000,
        "data/carclaims/attachments/reports", "data/carclaims/attachments/photos", carclaims_report, damage_photo,
        sources=("policyholder_age_group", "vehicle_category", "vehicle_price", "vehicle_age", "accident_rural",
                 "fault_policy_holder", "witness_present", "n_supplements"),
        forbidden=("fraud_found", "police_report_filed", "agent_internal", "deductible_100usd"),  # outcome + levers
    ),
    "illinois_wellness": DomainAttachments(
        "illinois_wellness", "employee_id", "data/illinois_wellness/wellness.csv", None, None,
        "data/illinois_wellness/attachments/reports", "data/illinois_wellness/attachments/charts", wellness_summary, activity_chart,
        sources=("male", "age50", "age37_49", "sickleave_0815_0716", "gym_0815_0716"),
        forbidden=("terminated_0119", "treat"),
    ),
    "german_credit": DomainAttachments(
        "german_credit", "applicant_id", "data/german_credit/credit.csv", None, None,
        "data/german_credit/attachments/summaries", "data/german_credit/attachments/charts", loan_summary, profile_chart,
        sources=("purpose", "housing", "employment_since", "savings", "age"),
        forbidden=("default", "duration_months", "credit_amount_kdm", "installment_rate"),  # outcome + levers
    ),
}


def write_pdf(lines: list[str], path: Path) -> None:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=11)
    for line in lines:
        pdf.multi_cell(0, 7, line, new_x="LMARGIN", new_y="NEXT")
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


def write_image(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, quality=90)


def rows_for(spec: DomainAttachments, repo_root: Path, limit: int | None = None) -> pd.DataFrame:
    """The rows that get attachments: the whole source file, or a fixed sample of it."""
    df = pd.read_csv(repo_root / spec.source)
    if spec.sample_size and len(df) > spec.sample_size:
        df = df.sample(spec.sample_size, random_state=7).sort_values(spec.id_column).reset_index(drop=True)
    return df.head(limit) if limit else df


def generate(domain_id: str, repo_root: Path, limit: int | None = None, seed: int = SEED, out_root: Path | None = None) -> dict:
    """Render every record's document and image; write the row sample if the domain uses one.
    Returns counts. `out_root` redirects the output tree (tests write under a temp dir)."""
    spec = DOMAINS[domain_id]
    base = out_root or repo_root
    df = rows_for(spec, repo_root, limit)
    for _, row in df.iterrows():
        rid = int(row[spec.id_column])
        rng = _rng(seed, rid)
        write_pdf(spec.document(row, rng), base / spec.pdf_dir / f"{rid}.pdf")
        write_image(spec.image(row, rng), base / spec.image_dir / f"{rid}.jpg")
    if spec.sample_output:
        path = base / spec.sample_output
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return {"domain": domain_id, "records": len(df), "pdf_dir": str(base / spec.pdf_dir), "image_dir": str(base / spec.image_dir)}
