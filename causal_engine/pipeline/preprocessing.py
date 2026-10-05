"""Stage 2 preprocessing: categorical encoding and missing-value handling.

Turns a domain's ingested frame into the all-numeric feature table the Feast
schema (Float32 features + Int64 outcome) and every downstream stage expect.
Driven entirely by the domain config's `feature_store` block:

    feature_store:
      entity_id_column: id
      feature_columns: [age, income]          # already numeric
      categorical_columns:
        - {name: checking, encoding: ordinal, order: [A11, A12, A13, A14]}
        - {name: foreign_worker, encoding: binary, positive: A201}
        - {name: housing, encoding: onehot}   # -> housing__A152, housing__A153, ...
      missing:
        numeric: median      # median | mean | drop | error (default: error)
        add_indicator: true  # also emit <col>__missing (0/1) for columns that had gaps

Missing data is a modelling decision (imputing or dropping rows can bias a
causal estimate), so the default is to fail loudly rather than guess.
Ordinal/binary columns follow the `missing.numeric` strategy after encoding;
one-hot columns get an explicit `missing` level instead. Rows with a missing
*outcome* never reach this stage -- ingestion drops them.

Output column names are predictable so other config sections can reference
them: ordinal/binary keep the source column's name, one-hot columns are
`<col>__<level>` (non-word characters in the level become `_`), indicators
are `<col>__missing`. The first one-hot level (or the first of `levels`) is
the dropped reference category unless `drop_first: false`, which avoids the
perfect collinearity that breaks the linear estimators.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import pandas as pd

from causal_engine.pipeline import modalities

logger = logging.getLogger(__name__)

NUMERIC_STRATEGIES = ("median", "mean", "drop", "error")
ENCODINGS = ("ordinal", "binary", "onehot")
MISSING_LEVEL = "missing"


@dataclass(frozen=True)
class PreprocessResult:
    df: pd.DataFrame  # entity id + encoded feature columns + outcome, clean index
    feature_columns: list[str]  # encoded feature columns, in output order
    report: dict


def feature_columns_of(feature_df: pd.DataFrame, domain_config: dict) -> list[str]:
    """Encoded feature columns of a Stage 2 frame: everything but the entity id
    and outcome. Use this instead of `feature_store.feature_columns`, which lists
    the raw columns and misses one-hot / indicator columns."""
    id_col = domain_config["feature_store"]["entity_id_column"]
    outcome_col = domain_config["ingestion"]["outcome_column"]
    return [c for c in feature_df.columns if c not in (id_col, outcome_col)]


def _level_name(level: str) -> str:
    return re.sub(r"\W+", "_", level).strip("_")


def _as_numeric(series: pd.Series, col: str) -> pd.Series:
    try:
        return pd.to_numeric(series, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"feature column {col!r} is not numeric ({exc}); declare it under "
            "feature_store.categorical_columns to encode it"
        ) from exc


def _encode_ordinal(s: pd.Series, cfg: dict) -> pd.Series:
    order = [str(v) for v in cfg["order"]]
    unknown = sorted(set(s.dropna()) - set(order))
    if unknown:
        raise ValueError(f"column {cfg['name']!r}: values {unknown} not in order {order}")
    return s.map({v: float(i) for i, v in enumerate(order)}).astype(float)


def _encode_binary(s: pd.Series, cfg: dict) -> pd.Series:
    out = (s == str(cfg["positive"])).astype(float)
    return out.where(s.notna())  # keep gaps as NaN for the missing-value step


def _encode_onehot(s: pd.Series, cfg: dict) -> dict[str, pd.Series]:
    col = cfg["name"]
    observed = sorted(set(s.dropna()))
    levels = [str(v) for v in cfg["levels"]] if cfg.get("levels") else observed
    unknown = sorted(set(observed) - set(levels))
    if unknown:
        raise ValueError(f"column {col!r}: values {unknown} not in levels {levels}")
    if s.isna().any():
        if MISSING_LEVEL in levels:
            raise ValueError(f"column {col!r} has a real level named {MISSING_LEVEL!r}")
        levels = levels + [MISSING_LEVEL]
        s = s.fillna(MISSING_LEVEL)
    reference = levels[0] if cfg.get("drop_first", True) else None
    return {
        f"{col}__{_level_name(lvl)}": (s == lvl).astype(float)
        for lvl in levels
        if lvl != reference
    }


def preprocess(df: pd.DataFrame, domain_config: dict) -> PreprocessResult:
    fs_cfg = domain_config["feature_store"]
    id_col = fs_cfg["entity_id_column"]
    outcome_col = domain_config["ingestion"]["outcome_column"]
    # attachment features (PDF / image columns from Stage 1) are numeric features like any other
    numeric_cols = list(fs_cfg["feature_columns"]) + modalities.attachment_columns(domain_config)
    cat_cfgs = fs_cfg.get("categorical_columns", [])
    missing_cfg = fs_cfg.get("missing", {})
    strategy = missing_cfg.get("numeric", "error")
    add_indicator = bool(missing_cfg.get("add_indicator", False))

    if strategy not in NUMERIC_STRATEGIES:
        raise ValueError(f"missing.numeric={strategy!r}; expected one of {NUMERIC_STRATEGIES}")
    for cfg in cat_cfgs:
        if cfg.get("encoding") not in ENCODINGS:
            raise ValueError(
                f"column {cfg.get('name')!r}: encoding={cfg.get('encoding')!r}; "
                f"expected one of {ENCODINGS}"
            )
    needed = [id_col, outcome_col, *numeric_cols, *(c["name"] for c in cat_cfgs)]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        raise ValueError(f"data is missing configured column(s) {absent}")

    # Columns that are numeric once encoded, and so go through the missing step.
    numeric_like: dict[str, pd.Series] = {c: _as_numeric(df[c], c) for c in numeric_cols}
    onehot_cols: dict[str, pd.Series] = {}
    encoded_from: dict[str, list[str]] = {}
    for cfg in cat_cfgs:
        name = cfg["name"]
        s = df[name].astype("string")
        if cfg["encoding"] == "ordinal":
            numeric_like[name] = _encode_ordinal(s, cfg)
            encoded_from[name] = [name]
        elif cfg["encoding"] == "binary":
            numeric_like[name] = _encode_binary(s, cfg)
            encoded_from[name] = [name]
        else:
            expanded = _encode_onehot(s, cfg)
            onehot_cols.update(expanded)
            encoded_from[name] = list(expanded)

    features = pd.DataFrame({**numeric_like, **onehot_cols}, index=df.index)
    if features.columns.duplicated().any() or {id_col, outcome_col} & set(features.columns):
        raise ValueError(f"encoded feature names collide: {list(features.columns)}")

    gaps = features[list(numeric_like)].isna().sum()
    gaps = {c: int(n) for c, n in gaps.items() if n}
    keep = pd.Series(True, index=df.index)
    if gaps:
        if strategy == "error":
            raise ValueError(
                f"missing values in feature column(s) {gaps}; set "
                "feature_store.missing.numeric to median, mean or drop"
            )
        if add_indicator and strategy != "drop":  # all zeros after dropping
            for col in gaps:
                features[f"{col}__missing"] = features[col].isna().astype(float)
        if strategy == "drop":
            keep = ~features[list(gaps)].isna().any(axis=1)
        else:
            fill = getattr(features[list(gaps)], strategy)()
            features = features.fillna(fill)

    out = features.assign(**{id_col: df[id_col], outcome_col: df[outcome_col]})[keep]
    out = out.reset_index(drop=True)
    feature_columns = list(features.columns)
    out = out[[id_col, *feature_columns, outcome_col]]

    report = {
        "rows_in": len(df),
        "rows_out": len(out),
        "rows_dropped_missing_features": int((~keep).sum()),
        "missing_strategy": strategy if gaps else None,
        "missing_counts": gaps,
        "encoded_from": encoded_from,
    }
    if gaps:
        logger.warning("preprocessing handled missing feature values: %s", report)
    return PreprocessResult(df=out, feature_columns=feature_columns, report=report)
