"""Stage 1 attachments: turn PDFs and images into numeric feature columns.

A domain's `ingestion.attachments` names, per dataset kind, files that sit beside the
tabular rows (one file per record id) and how to read them:

    ingestion:
      attachments:
        multimodal:
          - name: claim_report            # column prefix: claim_report__n_words, ...
            kind: pdf
            directory: data/carclaims/attachments/reports
            pattern: "{id}.pdf"           # file name for a record id
            keywords:                     # group -> substrings counted (case-insensitive)
              police_filed: ["a police report was filed"]
          - name: damage_photo
            kind: image
            directory: data/carclaims/attachments/photos
            pattern: "{id}.jpg"
            model: resnet18               # resnet18 | vit_b_16 (torchvision, ImageNet weights)
            embedding_dims: 8             # PCA components kept as <name>__cnn1..cnn8

PDFs: text is extracted with pypdf, then summarised as word/character counts, the
share of digits, and the count of each configured keyword group. Images: Pillow loads
and resizes, OpenCV computes brightness, contrast, edge density and sharpness, and a
pretrained torchvision network gives an embedding that PCA reduces to `embedding_dims`
columns. A record with no file gets NaN in every column of that attachment (Stage 2's
`feature_store.missing` policy then decides what to do; the default is to fail loudly).

What this does and does not do. The embeddings are generic ImageNet features, not
trained on the domain: they capture texture, colour and layout, and whether that is
useful for a causal question is for the domain to show. PCA is fitted on the rows being
ingested (unsupervised, no outcome), so the columns of one run are not comparable with
those of another. Optional dependencies (pypdf, Pillow, opencv-python-headless, torch,
torchvision) are imported only when an attachment of that kind is configured.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
KINDS = ("pdf", "image")
MODELS = ("resnet18", "vit_b_16")
IMAGE_SIZE = 224
DEFAULT_EMBEDDING_DIMS = 8
DEFAULT_BATCH = 64
PDF_BASE_FEATURES = ("n_words", "n_chars", "digit_share")
IMAGE_STAT_FEATURES = ("brightness", "contrast", "edge_density", "sharpness")


def _require(module: str, extra: str):
    try:
        return __import__(module, fromlist=["_"])
    except ImportError as exc:
        raise ImportError(f"attachments of this kind need `{extra}` (pip install {extra})") from exc


@dataclass(frozen=True)
class AttachmentSpec:
    name: str
    kind: str
    directory: Path
    pattern: str
    keywords: dict[str, list[str]]
    model: str
    embedding_dims: int
    image_size: int

    @classmethod
    def from_config(cls, raw: dict, repo_root: Path) -> "AttachmentSpec":
        kind = raw.get("kind")
        if kind not in KINDS:
            raise ValueError(f"attachment {raw.get('name')!r}: kind={kind!r}; expected one of {list(KINDS)}")
        if not raw.get("name") or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", raw["name"]):
            raise ValueError(f"attachment name {raw.get('name')!r} must be an identifier (letters, digits, underscores)")
        pattern = raw.get("pattern", "{id}")
        if "{id}" not in pattern:
            raise ValueError(f"attachment {raw['name']!r}: pattern {pattern!r} must contain {{id}}")
        model = raw.get("model", "resnet18")
        if kind == "image" and model not in MODELS:
            raise ValueError(f"attachment {raw['name']!r}: model={model!r}; expected one of {list(MODELS)}")
        dims = int(raw.get("embedding_dims", DEFAULT_EMBEDDING_DIMS))
        if dims < 0:
            raise ValueError(f"attachment {raw['name']!r}: embedding_dims must be >= 0")
        directory = Path(raw["directory"])
        return cls(
            name=raw["name"],
            kind=kind,
            directory=directory if directory.is_absolute() else repo_root / directory,
            pattern=pattern,
            keywords={k: [t.lower() for t in v] for k, v in (raw.get("keywords") or {}).items()},
            model=model,
            embedding_dims=dims,
            image_size=int(raw.get("image_size", IMAGE_SIZE)),
        )

    def path_for(self, record_id) -> Path:
        return self.directory / self.pattern.format(id=record_id)

    def feature_names(self) -> list[str]:
        """The columns this attachment adds, in order."""
        if self.kind == "pdf":
            names = [*PDF_BASE_FEATURES, *(f"kw_{group}" for group in self.keywords)]
        else:
            names = [*IMAGE_STAT_FEATURES, *(f"cnn{i}" for i in range(1, self.embedding_dims + 1))]
        return [f"{self.name}__{n}" for n in names]


EVAL_TAG_PREFIX = "eval_"  # the evaluation harness tags Feast state as eval_<kind> (harness.EVAL_DATASET_PREFIX)


def dataset_kind(domain_config: dict) -> str:
    """The dataset kind this run was tagged with (`run.dataset`, minus the harness's `eval_`
    prefix), else the domain's default."""
    tag = (domain_config.get("run") or {}).get("dataset") or domain_config["ingestion"].get("default_dataset", "")
    return tag[len(EVAL_TAG_PREFIX):] if tag.startswith(EVAL_TAG_PREFIX) else tag


def active_specs(domain_config: dict, repo_root: Path = REPO_ROOT) -> list[AttachmentSpec]:
    """Attachment specs that apply to this run's dataset kind (empty for most runs)."""
    by_kind = domain_config["ingestion"].get("attachments") or {}
    return [AttachmentSpec.from_config(raw, repo_root) for raw in by_kind.get(dataset_kind(domain_config), [])]


def attachment_columns(domain_config: dict, repo_root: Path = REPO_ROOT) -> list[str]:
    return [c for spec in active_specs(domain_config, repo_root) for c in spec.feature_names()]


# --- PDF ----------------------------------------------------------------------------------


def extract_pdf_text(path: Path) -> str:
    pypdf = _require("pypdf", "pypdf")
    reader = pypdf.PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def pdf_features(text: str, spec: AttachmentSpec) -> dict[str, float]:
    lowered = text.lower()
    chars = [c for c in text if not c.isspace()]
    features = {
        "n_words": float(len(text.split())),
        "n_chars": float(len(chars)),
        "digit_share": float(sum(c.isdigit() for c in chars) / len(chars)) if chars else 0.0,
    }
    for group, terms in spec.keywords.items():
        features[f"kw_{group}"] = float(sum(lowered.count(t) for t in terms))
    return features


# --- images -------------------------------------------------------------------------------


def load_image(path: Path, size: int = IMAGE_SIZE) -> np.ndarray:
    """RGB uint8 array of shape (size, size, 3)."""
    Image = _require("PIL.Image", "Pillow")
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((size, size)))


def image_stats(rgb: np.ndarray) -> dict[str, float]:
    """Brightness, contrast, edge density (share of Canny edge pixels) and sharpness (variance
    of the Laplacian) of the grey-scale image."""
    cv2 = _require("cv2", "opencv-python-headless")
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "edge_density": float((edges > 0).mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }


class CnnEmbedder:
    """A pretrained torchvision network as a fixed feature extractor (ImageNet weights, eval mode,
    classifier head removed): resnet18 gives 512 values per image, vit_b_16 gives 768."""

    def __init__(self, model_name: str = "resnet18"):
        if model_name not in MODELS:
            raise ValueError(f"model={model_name!r}; expected one of {list(MODELS)}")
        torch = _require("torch", "torch torchvision")
        models = _require("torchvision.models", "torch torchvision")
        self.torch = torch
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT
            net = models.resnet18(weights=weights)
            net.fc = torch.nn.Identity()
        else:
            weights = models.ViT_B_16_Weights.DEFAULT
            net = models.vit_b_16(weights=weights)
            net.heads = torch.nn.Identity()
        self.model_name = model_name
        self.net = net.eval()
        self.transform = weights.transforms(antialias=True)

    def embed(self, images: list[np.ndarray], batch_size: int = DEFAULT_BATCH) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = np.stack(images[start : start + batch_size])  # (b, h, w, 3) uint8
                tensor = torch.from_numpy(batch).permute(0, 3, 1, 2)
                out.append(self.net(self.transform(tensor)).cpu().numpy())
        return np.concatenate(out) if out else np.empty((0, 0))


def reduce_embeddings(embeddings: np.ndarray, dims: int, seed: int = 0) -> np.ndarray:
    """PCA to `dims` components (fewer if there are fewer rows than that). Returns (n, dims),
    zero-padded when there are too few rows, so the column set never depends on the data size."""
    from sklearn.decomposition import PCA

    n, width = embeddings.shape
    keep = min(dims, n, width)
    reduced = PCA(n_components=keep, random_state=seed).fit_transform(embeddings) if keep else np.empty((n, 0))
    if keep < dims:
        reduced = np.hstack([reduced, np.zeros((n, dims - keep))])
    return reduced


# --- one attachment -> a feature frame ----------------------------------------------------


def extract_features(ids: pd.Series, spec: AttachmentSpec, embedder: Optional[CnnEmbedder] = None) -> pd.DataFrame:
    """Feature columns for `spec`, one row per id, in `ids` order. A missing or unreadable file
    gives NaN in every column of that row."""
    columns = spec.feature_names()
    frame = pd.DataFrame(np.nan, index=range(len(ids)), columns=columns)
    unreadable = 0
    present: list[int] = []
    arrays: list[np.ndarray] = []

    for position, record_id in enumerate(ids):
        path = spec.path_for(record_id)
        if not path.is_file():
            continue
        try:
            if spec.kind == "pdf":
                features = pdf_features(extract_pdf_text(path), spec)
                frame.loc[position, [f"{spec.name}__{k}" for k in features]] = list(features.values())
            else:
                rgb = load_image(path, spec.image_size)
                stats = image_stats(rgb)
                frame.loc[position, [f"{spec.name}__{k}" for k in stats]] = list(stats.values())
                arrays.append(rgb)
                present.append(position)
        except Exception as exc:  # a corrupt file must not stop a run; it becomes a missing value
            unreadable += 1
            logger.warning("attachment %s: could not read %s (%s)", spec.name, path, exc)

    if spec.kind == "image" and spec.embedding_dims and arrays:
        embedder = embedder or CnnEmbedder(spec.model)
        reduced = reduce_embeddings(embedder.embed(arrays), spec.embedding_dims)
        frame.loc[present, [f"{spec.name}__cnn{i}" for i in range(1, spec.embedding_dims + 1)]] = reduced

    frame.index = ids.index
    n_missing = int(frame.isna().all(axis=1).sum())
    frame.attrs["report"] = {
        "name": spec.name, "kind": spec.kind, "rows": len(ids), "missing_files": n_missing - unreadable, "unreadable": unreadable,
    }
    return frame


def attach_features(df: pd.DataFrame, domain_config: dict, repo_root: Path = REPO_ROOT) -> pd.DataFrame:
    """`df` with every active attachment's feature columns appended (a no-op when none apply)."""
    specs = active_specs(domain_config, repo_root)
    if not specs:
        return df
    id_col = domain_config["ingestion"]["id_column"]
    out = df
    reports = []
    for spec in specs:
        overlap = [c for c in spec.feature_names() if c in out.columns]
        if overlap:
            raise ValueError(f"attachment {spec.name!r} would overwrite existing columns {overlap}")
        features = extract_features(out[id_col], spec)
        reports.append(features.attrs["report"])
        out = pd.concat([out, features], axis=1)
    out.attrs.update(df.attrs)
    out.attrs["attachment_report"] = reports
    for report in reports:
        logger.info("attachments: %s", report)
    return out
