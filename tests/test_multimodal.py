"""PDF / image attachments (causal_engine/pipeline/modalities.py) and the synthetic renderers behind them
(causal_engine/evaluation/attachment_synth.py).

Rendering and extraction run for real: fpdf2 writes PDFs that pypdf reads back, Pillow/OpenCV process
real pixels. The CNN tests need torch/torchvision and the pretrained weights (downloaded once); they are
skipped, not failed, where those are unavailable.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from causal_engine.evaluation import attachment_synth, harness
from causal_engine.pipeline import effect_estimation, ingestion, modalities, preprocessing
from causal_engine.utils.config_loader import ConfigLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
DOMAINS = sorted(attachment_synth.DOMAINS)


def _rows(domain: str, n: int = 24) -> pd.DataFrame:
    spec = attachment_synth.DOMAINS[domain]
    return pd.read_csv(REPO_ROOT / spec.source).head(n)


@pytest.fixture(scope="module")
def carclaims_attachments(tmp_path_factory):
    """25 rendered carclaims records under a temp tree, and a config pointing at them."""
    for module in ("fpdf", "pypdf", "PIL", "cv2"):  # requirements-multimodal.txt; skip without them
        pytest.importorskip(module)
    out = tmp_path_factory.mktemp("attachments")
    attachment_synth.generate("carclaims", REPO_ROOT, limit=25, out_root=out)
    cfg = copy.deepcopy(ConfigLoader().get_domain("carclaims").extra)
    cfg["ingestion"]["datasets"]["multimodal"] = str(out / "data/carclaims/claims_multimodal.csv")
    for spec in cfg["ingestion"]["attachments"]["multimodal"]:
        spec["directory"] = str(out / spec["directory"])
        if spec["kind"] == "image":
            spec["embedding_dims"] = 0  # no network model needed for the plumbing tests
    cfg["run"] = {"dataset": "multimodal"}
    return cfg, out


# --- renderers ----------------------------------------------------------------------------


@pytest.mark.parametrize("domain", DOMAINS)
def test_rendering_is_deterministic_and_differs_between_records(domain):
    spec, rows = attachment_synth.DOMAINS[domain], _rows(domain, 4)
    first, second = rows.iloc[0], rows.iloc[1]
    a = spec.document(first, attachment_synth._rng(7, first[spec.id_column]))
    b = spec.document(first, attachment_synth._rng(7, first[spec.id_column]))
    other = spec.document(second, attachment_synth._rng(7, second[spec.id_column]))
    assert a == b and a != other
    image = spec.image(first, attachment_synth._rng(7, first[spec.id_column]))
    again = spec.image(first, attachment_synth._rng(7, first[spec.id_column]))
    assert image.shape == (224, 224, 3) and image.dtype == np.uint8 and np.array_equal(image, again)


@pytest.mark.parametrize("domain", DOMAINS)
def test_attachments_never_depend_on_the_outcome_or_the_randomized_treatment(domain):
    """The no-leakage rule: flipping every forbidden column (outcome, treatment) changes nothing rendered."""
    spec, rows = attachment_synth.DOMAINS[domain], _rows(domain, 8)
    flipped = rows.copy()
    for column in spec.forbidden:
        flipped[column] = 1 - flipped[column]
    assert not flipped[list(spec.forbidden)].equals(rows[list(spec.forbidden)])  # the test has teeth
    for (_, row), (_, changed) in zip(rows.iterrows(), flipped.iterrows()):
        rid = row[spec.id_column]
        assert spec.document(row, attachment_synth._rng(3, rid)) == spec.document(changed, attachment_synth._rng(3, rid))
        assert np.array_equal(spec.image(row, attachment_synth._rng(3, rid)), spec.image(changed, attachment_synth._rng(3, rid)))


@pytest.mark.parametrize("domain", DOMAINS)
def test_no_stage_4_treatment_or_outcome_is_ever_rendered_into_an_attachment(domain):
    """Rendering a lever would put a copy of the treatment among the features (a bad control);
    rendering the outcome would leak it. Checked against the domain's own config."""
    cfg = ConfigLoader().get_domain(domain).extra
    spec = attachment_synth.DOMAINS[domain]
    treatments = {t["name"] for t in cfg["effect_estimation"]["treatments"]}
    treatments |= {"treat"} & set(cfg["feature_store"]["feature_columns"])  # a randomized treatment listed as a feature
    outcome = cfg["ingestion"]["outcome_column"]
    assert (treatments | {outcome}) <= set(spec.forbidden)
    assert not (treatments | {outcome}) & set(spec.sources)


def test_every_forbidden_column_is_a_real_outcome_or_treatment_and_no_source_is_forbidden():
    for domain, spec in attachment_synth.DOMAINS.items():
        assert not set(spec.sources) & set(spec.forbidden), domain
        columns = set(pd.read_csv(REPO_ROOT / spec.source, nrows=1).columns)
        assert set(spec.forbidden) <= columns and set(spec.sources) <= columns, domain
    assert "treat" in attachment_synth.DOMAINS["illinois_wellness"].forbidden  # the randomized treatment is off limits too


def test_more_damage_means_more_edges_in_the_photo():
    pytest.importorskip("PIL")
    pytest.importorskip("cv2")
    row = _rows("carclaims", 1).iloc[0]
    edges = {}
    for level in ("none", "gt_5"):
        rows = row.copy()
        rows["n_supplements"] = level
        edges[level] = np.mean(
            [modalities.image_stats(attachment_synth.damage_photo(rows, attachment_synth._rng(s, 1)))["edge_density"] for s in range(12)]
        )
    assert edges["gt_5"] > edges["none"] * 1.1


def test_carclaims_sample_is_fixed_and_smaller_than_the_full_file():
    spec = attachment_synth.DOMAINS["carclaims"]
    sample = attachment_synth.rows_for(spec, REPO_ROOT)
    assert len(sample) == 5000 and sample["claim_id"].is_unique
    assert list(sample["claim_id"]) == sorted(sample["claim_id"])
    pd.testing.assert_frame_equal(sample, attachment_synth.rows_for(spec, REPO_ROOT))  # same sample every time
    assert len(attachment_synth.rows_for(attachment_synth.DOMAINS["german_credit"], REPO_ROOT)) == 1000  # small: all rows


def test_generate_writes_one_pdf_and_one_image_per_record_and_the_row_sample(carclaims_attachments):
    cfg, out = carclaims_attachments
    reports = sorted((out / "data/carclaims/attachments/reports").glob("*.pdf"))
    photos = sorted((out / "data/carclaims/attachments/photos").glob("*.jpg"))
    sample = pd.read_csv(out / "data/carclaims/claims_multimodal.csv")
    assert len(reports) == len(photos) == len(sample) == 25
    assert {p.stem for p in reports} == {str(i) for i in sample["claim_id"]}


# --- extraction ---------------------------------------------------------------------------


def test_pdf_text_round_trips_through_pypdf(carclaims_attachments):
    cfg, out = carclaims_attachments
    sample = pd.read_csv(out / "data/carclaims/claims_multimodal.csv")
    row = sample.iloc[0]
    text = modalities.extract_pdf_text(out / f"data/carclaims/attachments/reports/{int(row['claim_id'])}.pdf")
    assert f"CLAIM REPORT {int(row['claim_id'])}" in text
    assert ("A witness was present." if row["witness_present"] else "No witness was present.") in text
    assert "police" not in text.lower()  # the police report is a Stage 4 lever: never rendered


def test_keyword_features_agree_with_the_fields_the_text_was_rendered_from(carclaims_attachments):
    cfg, out = carclaims_attachments
    ingested = ingestion.ingest(cfg["ingestion"]["datasets"]["multimodal"], cfg)
    assert ((ingested["claim_report__kw_fault"] > 0) == (ingested["fault_policy_holder"] == 1)).all()
    assert ((ingested["claim_report__kw_rural"] > 0) == (ingested["accident_rural"] == 1)).all()
    assert ((ingested["claim_report__kw_witness"] > 0) == (ingested["witness_present"] == 1)).all()
    assert (ingested["claim_report__n_words"] > 30).all()


def test_pdf_features_count_case_insensitively_and_handle_empty_text():
    spec = modalities.AttachmentSpec.from_config(
        {"name": "doc", "kind": "pdf", "directory": "x", "keywords": {"police": ["Police Report"], "none": ["nothing here"]}}, REPO_ROOT
    )
    features = modalities.pdf_features("A police report. Another POLICE REPORT here 42", spec)
    assert features["kw_police"] == 2 and features["kw_none"] == 0 and features["n_words"] == 8
    assert features["digit_share"] == pytest.approx(2 / len("Apolicereport.AnotherPOLICEREPORThere42"))
    empty = modalities.pdf_features("", spec)
    assert empty["n_words"] == 0 and empty["digit_share"] == 0.0


def test_image_stats_on_known_images():
    pytest.importorskip("cv2")
    black = np.zeros((64, 64, 3), dtype=np.uint8)
    checker = (np.indices((64, 64)).sum(axis=0) // 8 % 2 * 255).astype(np.uint8)
    checker = np.stack([checker] * 3, axis=-1)
    flat, busy = modalities.image_stats(black), modalities.image_stats(checker)
    assert flat["brightness"] == 0 and flat["edge_density"] == 0 and flat["sharpness"] == 0
    assert busy["edge_density"] > 0.02 and busy["sharpness"] > 1000 and busy["contrast"] > 100


def test_missing_and_unreadable_files_become_nan_and_are_reported(carclaims_attachments, tmp_path):
    cfg, out = carclaims_attachments
    spec = modalities.AttachmentSpec.from_config(cfg["ingestion"]["attachments"]["multimodal"][0], REPO_ROOT)
    ids = pd.Series([int(p.stem) for p in sorted(spec.directory.glob("*.pdf"))[:2]] + [999_999_999, 888_888_888])
    bad = tmp_path / "junk"
    bad.mkdir()
    (bad / "888888888.pdf").write_bytes(b"this is not a pdf")
    broken = modalities.AttachmentSpec(**{**spec.__dict__, "directory": bad})
    assert modalities.extract_features(ids, spec).attrs["report"]["missing_files"] == 2  # two ids have no file

    frame = modalities.extract_features(pd.Series([888888888]), broken)
    assert frame.isna().all().all()
    assert frame.attrs["report"]["unreadable"] == 1 and frame.attrs["report"]["missing_files"] == 0

    features = modalities.extract_features(ids, spec)
    assert features.iloc[:2].notna().all().all() and features.iloc[2:].isna().all().all()
    assert list(features.index) == list(ids.index)  # row order and index preserved


def test_reduce_embeddings_keeps_the_column_count_even_with_few_rows():
    rng = np.random.default_rng(0)
    full = modalities.reduce_embeddings(rng.normal(size=(40, 30)), 8)
    assert full.shape == (40, 8) and np.allclose(full.mean(axis=0), 0, atol=1e-9)
    few = modalities.reduce_embeddings(rng.normal(size=(3, 30)), 8)
    assert few.shape == (3, 8) and np.all(few[:, 3:] == 0)  # only min(rows, dims) real components, then zeros
    assert modalities.reduce_embeddings(rng.normal(size=(5, 4)), 0).shape == (5, 0)


# --- spec validation ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,message",
    [
        ({"name": "a", "kind": "audio", "directory": "x"}, "kind='audio'"),
        ({"name": "1bad", "kind": "pdf", "directory": "x"}, "must be an identifier"),
        ({"name": "a", "kind": "pdf", "directory": "x", "pattern": "file.pdf"}, r"must contain \{id\}"),
        ({"name": "a", "kind": "image", "directory": "x", "model": "vgg"}, "model='vgg'"),
        ({"name": "a", "kind": "image", "directory": "x", "embedding_dims": -1}, "embedding_dims"),
    ],
)
def test_bad_attachment_specs_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        modalities.AttachmentSpec.from_config(raw, REPO_ROOT)


def test_feature_names_are_predictable():
    pdf = modalities.AttachmentSpec.from_config({"name": "r", "kind": "pdf", "directory": "x", "keywords": {"a": ["x"], "b": ["y"]}}, REPO_ROOT)
    assert pdf.feature_names() == ["r__n_words", "r__n_chars", "r__digit_share", "r__kw_a", "r__kw_b"]
    image = modalities.AttachmentSpec.from_config({"name": "p", "kind": "image", "directory": "x", "embedding_dims": 3}, REPO_ROOT)
    assert image.feature_names() == ["p__brightness", "p__contrast", "p__edge_density", "p__sharpness", "p__cnn1", "p__cnn2", "p__cnn3"]


# --- config, ingestion, stages ------------------------------------------------------------


def test_dataset_kind_comes_from_the_run_tag_and_ignores_the_harness_prefix():
    cfg = {"ingestion": {"default_dataset": "real"}, "run": {"dataset": "multimodal"}}
    assert modalities.dataset_kind(cfg) == "multimodal"
    assert modalities.dataset_kind({**cfg, "run": {"dataset": "eval_multimodal"}}) == "multimodal"
    assert modalities.dataset_kind({"ingestion": {"default_dataset": "real"}}) == "real"
    assert modalities.EVAL_TAG_PREFIX == harness.EVAL_DATASET_PREFIX  # the two must agree


def test_only_the_multimodal_dataset_has_attachments_and_others_are_untouched():
    cfg = ConfigLoader().get_domain("carclaims").extra
    assert modalities.active_specs(cfg) == []  # default dataset is real
    tagged = {**cfg, "run": {"dataset": "multimodal"}}
    assert [s.name for s in modalities.active_specs(tagged)] == ["claim_report", "damage_photo"]
    assert modalities.attachment_columns({**cfg, "run": {"dataset": "real"}}) == []
    df = pd.DataFrame({"claim_id": [1]})
    assert modalities.attach_features(df, cfg) is df  # a no-op returns the same object


def test_ingestion_appends_attachment_columns_without_touching_the_table(carclaims_attachments):
    cfg, out = carclaims_attachments
    path = cfg["ingestion"]["datasets"]["multimodal"]
    plain = ingestion.ingest(path, {**cfg, "run": {"dataset": "real"}})
    with_att = ingestion.ingest(path, cfg)
    added = modalities.attachment_columns(cfg)
    assert list(with_att.columns) == list(plain.columns) + added
    pd.testing.assert_frame_equal(with_att[plain.columns], plain)
    assert with_att.attrs["attachment_report"][0]["name"] == "claim_report"
    assert with_att.attrs["rows_dropped_missing_outcome"] == 0


def test_preprocessing_treats_attachment_columns_as_features_and_names_them_when_missing(carclaims_attachments):
    cfg, out = carclaims_attachments
    with_att = ingestion.ingest(cfg["ingestion"]["datasets"]["multimodal"], cfg)
    features = preprocessing.preprocess(with_att, cfg).feature_columns
    assert set(modalities.attachment_columns(cfg)) <= set(features)
    with pytest.raises(ValueError, match="claim_report__n_words"):
        preprocessing.preprocess(with_att.drop(columns=["claim_report__n_words"]), cfg)


def test_a_configured_attachment_that_would_overwrite_a_column_is_an_error(carclaims_attachments):
    cfg, out = carclaims_attachments
    df = pd.read_csv(cfg["ingestion"]["datasets"]["multimodal"]).assign(claim_report__n_words=1)
    with pytest.raises(ValueError, match="would overwrite"):
        modalities.attach_features(df, cfg)


def test_adjust_for_attachments_puts_the_feature_columns_in_the_adjustment_set():
    rng = np.random.default_rng(0)
    n = 4000
    latent = rng.normal(size=n)  # the attachment measures a confounder of a -> y
    a = latent + rng.normal(size=n)
    y = 0.5 * a + 1.0 * latent + rng.normal(size=n)
    df = pd.DataFrame({"id": range(n), "a": a, "claim_report__n_words": latent, "y": y})
    cfg = {
        "ingestion": {"default_dataset": "multimodal", "attachments": {"multimodal": [{"name": "claim_report", "kind": "pdf", "directory": "x"}]}},
        "effect_estimation": {"outcome": "y", "treatments": [{"name": "a", "confounders": []}], "refutation_simulations": 1, "refutation": None},
    }
    plain = effect_estimation.estimate_effects(df, cfg)[0].ate
    cfg["effect_estimation"]["adjust_for_attachments"] = True
    adjusted = effect_estimation.estimate_effects(df, cfg)[0].ate
    assert plain > 0.8 and adjusted == pytest.approx(0.5, abs=0.05)  # without the column the estimate is confounded

    cfg["run"] = {"dataset": "real"}  # no attachments for this run: nothing extra to adjust for
    assert effect_estimation.estimate_effects(df, cfg)[0].ate == pytest.approx(plain)


def test_config_loader_checks_the_shape_of_attachments(tmp_path):
    from tests.test_config_loader import _write_domain  # the loader tests' own helper

    base = "  datasets:\n    real: a.csv\n  default_dataset: real\n"
    _write_domain(tmp_path, base + "  attachments:\n    multimodal:\n      - {name: a, kind: pdf, directory: d}\n")
    with pytest.raises(ValueError, match="names dataset 'multimodal', which is not in ingestion.datasets"):
        ConfigLoader(tmp_path)
    _write_domain(tmp_path, base + "  attachments:\n    real:\n      - {name: a, kind: pdf}\n")
    with pytest.raises(ValueError, match="needs name, kind and directory"):
        ConfigLoader(tmp_path)
    _write_domain(tmp_path, base + "  attachments:\n    real:\n      - {name: a, kind: pdf, directory: d}\n      - {name: a, kind: image, directory: e}\n")
    with pytest.raises(ValueError, match="must be unique"):
        ConfigLoader(tmp_path)


def test_the_slow_multimodal_dataset_is_skipped_by_default_and_never_bootstrapped(monkeypatch):
    seen = []
    monkeypatch.setattr(harness, "evaluate_dataset", lambda domain, cfg, dataset, *a, **k: seen.append((domain, dataset)) or [])
    harness.run_evaluation(domains=["carclaims"])
    assert seen == [("carclaims", "real")]
    seen.clear()
    harness.run_evaluation(domains=["carclaims"], datasets=["multimodal"])
    assert seen == [("carclaims", "multimodal")]


# --- the CNN ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def resnet():
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    try:
        return modalities.CnnEmbedder("resnet18")
    except Exception as exc:  # weights need a one-time download
        pytest.skip(f"ResNet18 weights unavailable: {exc}")


def test_resnet_embeds_images_to_512_dimensions_deterministically(resnet):
    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(3)]
    first, second = resnet.embed(images), resnet.embed(images)
    assert first.shape == (3, 512) and np.allclose(first, second, atol=1e-5)
    assert not np.allclose(first[0], first[1])
    assert resnet.embed([]).shape[0] == 0


def test_cnn_columns_are_added_and_carry_the_rendered_quantity(carclaims_attachments, resnet):
    cfg, out = carclaims_attachments
    spec_cfg = {**cfg["ingestion"]["attachments"]["multimodal"][1], "embedding_dims": 4}
    spec = modalities.AttachmentSpec.from_config(spec_cfg, REPO_ROOT)
    sample = pd.read_csv(cfg["ingestion"]["datasets"]["multimodal"])
    frame = modalities.extract_features(sample["claim_id"], spec, embedder=resnet)
    assert list(frame.columns) == spec.feature_names() and frame.notna().all().all()
    cnn = frame[[c for c in frame.columns if "cnn" in c]].to_numpy()
    assert cnn.shape == (len(sample), 4) and np.abs(cnn.mean(axis=0)).max() < 1e-4  # PCA output is centred


def test_unknown_model_name_is_rejected_before_any_import():
    with pytest.raises(ValueError, match="model='alexnet'"):
        modalities.CnnEmbedder("alexnet")


def test_stage_4_refuses_to_adjust_for_an_attachment_feature_that_copies_the_treatment():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.normal(size=n)
    df = pd.DataFrame({"id": range(n), "a": a, "claim_report__n_words": -a + 1e-6 * rng.normal(size=n), "y": a + rng.normal(size=n)})
    cfg = {
        "ingestion": {"default_dataset": "multimodal", "attachments": {"multimodal": [{"name": "claim_report", "kind": "pdf", "directory": "x"}]}},
        "effect_estimation": {
            "outcome": "y", "treatments": [{"name": "a"}], "refutation": None, "adjust_for_attachments": True,
        },
    }
    with pytest.raises(ValueError, match="almost a copy of treatment 'a'"):
        effect_estimation.estimate_effects(df, cfg)
    cfg["effect_estimation"]["adjust_for_attachments"] = False  # off: nothing is adjusted for, so no complaint
    assert effect_estimation.estimate_effects(df, cfg)[0].ate == pytest.approx(1.0, abs=0.15)


def test_vit_embeds_to_768_dimensions_when_its_weights_are_already_cached():
    """ViT-B/16 weights are 330 MB, so this only runs where they have been downloaded already."""
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch

    cached = list((Path(torch.hub.get_dir()) / "checkpoints").glob("vit_b_16-*.pth"))
    if not cached:
        pytest.skip("vit_b_16 weights are not cached; load one once with modalities.CnnEmbedder('vit_b_16')")
    embedder = modalities.CnnEmbedder("vit_b_16")
    rng = np.random.default_rng(1)
    images = [rng.integers(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(2)]
    out = embedder.embed(images)
    assert out.shape == (2, 768) and np.allclose(out, embedder.embed(images), atol=1e-4)
