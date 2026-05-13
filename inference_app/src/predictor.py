"""
Model Prediction Module

This module handles model loading and prediction for inference.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BiasPredictor:
    """
    Predicts biased agonism for GPCR ligands.

    .. deprecated:: Phase 5
       Use :class:`ModernBiasPredictor` (defined later in this module)
       which loads the new sklearn-Pipeline-based artifacts produced by the
       rewritten Stage 10 trainer (``data/processed/ml_models/``). This
       class is preserved only for backwards compatibility with the legacy
       ``results/models/<name>.pkl`` + separate scaler/imputer layout.
    """

    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        metadata_path: str,
        imputer_path: Optional[str] = None,
    ):
        """
        Initialize the predictor.

        Args:
            model_path: Path to trained model pickle file
            scaler_path: Path to scaler pickle file
            metadata_path: Path to preprocessing metadata JSON file
            imputer_path: Optional path to imputer pickle file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.metadata_path = metadata_path
        self.imputer_path = imputer_path

        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.label_mapping = None
        self.reverse_label_mapping = None

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load all required artifacts (model, scaler, metadata)."""
        import json

        logger.info("Loading prediction artifacts...")

        # Load model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        with open(self.model_path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Loaded model from: {self.model_path}")

        # Load scaler
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")

        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        logger.info(f"Loaded scaler from: {self.scaler_path}")

        # Load imputer if provided
        if self.imputer_path and os.path.exists(self.imputer_path):
            with open(self.imputer_path, "rb") as f:
                self.imputer = pickle.load(f)
            logger.info(f"Loaded imputer from: {self.imputer_path}")

        # Load metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.feature_columns = metadata.get("feature_columns", [])
        self.label_mapping = metadata.get("label_mapping", {})

        # Create reverse mapping (numeric -> label)
        if self.label_mapping:
            self.reverse_label_mapping = {v: k for k, v in self.label_mapping.items()}

        logger.info(f"Loaded metadata with {len(self.feature_columns)} features")
        logger.info(f"Label mapping: {self.label_mapping}")

    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features to match training data format.

        Args:
            features_df: DataFrame with molecular descriptors

        Returns:
            Preprocessed feature array ready for prediction
        """
        # Make a copy to avoid modifying the original
        features_df = features_df.copy()
        
        # Select only the features used during training
        if self.feature_columns:
            missing_features = [
                col for col in self.feature_columns if col not in features_df.columns
            ]

            if missing_features:
                logger.warning(
                    f"Missing {len(missing_features)} features. Filling with NaN."
                )
                logger.warning(f"Missing features: {missing_features}")
                # Add missing features as NaN
                for col in missing_features:
                    features_df[col] = np.nan

            # Select features in the correct order (keep as DataFrame for feature names)
            features_df = features_df[self.feature_columns].copy()

        # Handle missing values with imputer
        if self.imputer is not None:
            # Transform and keep as DataFrame to preserve feature names
            imputed_values = self.imputer.transform(features_df)
            features_df = pd.DataFrame(imputed_values, columns=self.feature_columns)
        else:
            # Simple median imputation if no imputer
            features_df = features_df.fillna(features_df.median())

        # Scale features - pass DataFrame to preserve feature names
        features_scaled = self.scaler.transform(features_df)

        return features_scaled

    def predict(
        self, features_df: pd.DataFrame, return_proba: bool = True
    ) -> Tuple[str, Dict[str, float]]:
        """
        Predict bias category for given features.

        Args:
            features_df: DataFrame with molecular descriptors
            return_proba: Whether to return class probabilities

        Returns:
            Tuple of (predicted_class, class_probabilities_dict)
        """
        # Preprocess features
        features_processed = self.preprocess_features(features_df)

        # Make prediction
        predicted_class_idx = self.model.predict(features_processed)[0]

        # Convert to class label
        if self.reverse_label_mapping:
            predicted_class = self.reverse_label_mapping.get(
                int(predicted_class_idx), f"Class_{predicted_class_idx}"
            )
        else:
            predicted_class = f"Class_{predicted_class_idx}"

        # Get probabilities if available
        probabilities = {}
        if return_proba and hasattr(self.model, "predict_proba"):
            proba_array = self.model.predict_proba(features_processed)[0]
            class_indices = self.model.classes_

            for idx, prob in zip(class_indices, proba_array):
                if self.reverse_label_mapping:
                    class_name = self.reverse_label_mapping.get(
                        int(idx), f"Class_{idx}"
                    )
                else:
                    class_name = f"Class_{idx}"
                probabilities[class_name] = float(prob)

        return predicted_class, probabilities

    def get_class_names(self) -> List[str]:
        """
        Get list of class names.

        Returns:
            List of class names
        """
        if self.label_mapping:
            return list(self.label_mapping.keys())
        return []


def load_predictor(
    model_name: str = "random_forest",
    base_path: str = None,
) -> BiasPredictor:
    """
    Convenience function to load a predictor with default paths.

    Args:
        model_name: Name of the model to load (without .pkl extension)
        base_path: Base path to project root (defaults to parent of inference_app)

    Returns:
        Initialized BiasPredictor instance
    """
    if base_path is None:
        # Default to parent directory of inference_app
        base_path = Path(__file__).parent.parent.parent

    base_path = Path(base_path)

    model_path = base_path / "results" / "models" / f"{model_name}.pkl"
    scaler_path = base_path / "data" / "processed" / "ml_preprocessed" / "scaler.pkl"
    metadata_path = (
        base_path
        / "data"
        / "processed"
        / "ml_preprocessed"
        / "preprocessing_metadata.json"
    )
    imputer_path = base_path / "data" / "processed" / "ml_preprocessed" / "imputer.pkl"

    # Check if imputer exists
    if not imputer_path.exists():
        imputer_path = None

    return BiasPredictor(
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        metadata_path=str(metadata_path),
        imputer_path=str(imputer_path) if imputer_path else None,
    )


def resolve_model_name(
    base_path: Path | str | None = None,
    *,
    fallback: str = "random_forest",
) -> str:
    """Pick the model name from the locked selection decision.

    Stage 12 fix — see improvements/12_inference_deployment.md F12.2.

    The legacy `load_predictor(model_name="random_forest")` hardcoded the
    chosen model in app code, so the manuscript's "we picked X" claim and
    the deployed model could disagree silently. This helper reads
    `results/model_selection_decision.json` (written by
    `cancerag.ml.model_selection.write_selection_decision`) and falls back
    to the supplied default only when the artifact is absent — preserving
    backwards-compatibility.
    """
    import json as _json

    if base_path is None:
        base_path = Path(__file__).parent.parent.parent
    base_path = Path(base_path)
    decision_path = base_path / "results" / "model_selection_decision.json"
    if not decision_path.exists():
        return fallback
    try:
        decision = _json.loads(decision_path.read_text())
    except Exception:
        return fallback
    chosen = decision.get("chosen")
    return chosen if isinstance(chosen, str) and chosen else fallback


# =====================================================================
# Stage 12 — applicability-domain check + /health-style fingerprint
# payload. Live here next to BiasPredictor / load_predictor / resolve_model_name
# so the inference app has one canonical predictor module.
# =====================================================================


import hashlib as _hashlib  # noqa: E402
import json as _json2  # noqa: E402  (json is already imported transitively below)
import platform as _platform  # noqa: E402
import subprocess as _subprocess  # noqa: E402
from dataclasses import dataclass as _ad_dc  # noqa: E402
from datetime import datetime as _datetime, timezone as _timezone  # noqa: E402
from typing import Any as _Any, Iterable, Sequence  # noqa: E402

from rdkit import Chem as _Chem, DataStructs as _DataStructs  # noqa: E402
from rdkit.Chem import AllChem as _AllChem  # noqa: E402


# ---------------------------------------------------- applicability domain


@_ad_dc(frozen=True)
class ADResult:
    in_domain: bool
    nearest_neighbor_tanimoto: float
    threshold: float


def _smiles_to_morgan_bv(smiles: str, *, radius: int = 2, n_bits: int = 2048):
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = _Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)


class ApplicabilityChecker:
    """Holds the training fingerprints and computes per-query AD results."""

    def __init__(
        self,
        train_smiles: Iterable[str],
        *,
        threshold: float = 0.4,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"threshold must be in [0, 1]; got {threshold!r}"
            )
        self.threshold = float(threshold)
        self.radius = int(radius)
        self.n_bits = int(n_bits)
        self._train_fps = []
        for s in train_smiles:
            bv = _smiles_to_morgan_bv(s, radius=radius, n_bits=n_bits)
            if bv is not None:
                self._train_fps.append(bv)
        if not self._train_fps:
            raise ValueError(
                "ApplicabilityChecker: no parseable training SMILES provided"
            )

    def __len__(self) -> int:
        return len(self._train_fps)

    def check(self, smiles: str) -> ADResult:
        bv = _smiles_to_morgan_bv(
            smiles, radius=self.radius, n_bits=self.n_bits
        )
        if bv is None:
            return ADResult(
                in_domain=False,
                nearest_neighbor_tanimoto=0.0,
                threshold=self.threshold,
            )
        sims = _DataStructs.BulkTanimotoSimilarity(bv, self._train_fps)
        nn = float(max(sims)) if sims else 0.0
        return ADResult(
            in_domain=nn >= self.threshold,
            nearest_neighbor_tanimoto=nn,
            threshold=self.threshold,
        )

    def batch_check(self, smiles_list: Sequence[str]) -> list[ADResult]:
        return [self.check(s) for s in smiles_list]


def nearest_neighbor_tanimoto(
    query_smiles: str,
    train_smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Stateless one-off helper — for ad-hoc CLI use."""
    bv = _smiles_to_morgan_bv(query_smiles, radius=radius, n_bits=n_bits)
    if bv is None:
        return 0.0
    train_fps = []
    for s in train_smiles:
        train_bv = _smiles_to_morgan_bv(s, radius=radius, n_bits=n_bits)
        if train_bv is not None:
            train_fps.append(train_bv)
    if not train_fps:
        return 0.0
    return float(max(_DataStructs.BulkTanimotoSimilarity(bv, train_fps)))


def confidence_label(ad: ADResult, max_class_proba: float) -> str:
    """Combine AD status and predicted-class probability into a UI-friendly
    confidence badge: 'high' / 'medium' / 'low' / 'out_of_domain'."""
    if not ad.in_domain:
        return "out_of_domain"
    if max_class_proba >= 0.7:
        return "high"
    if max_class_proba >= 0.5:
        return "medium"
    return "low"


def ad_to_dict(ad: ADResult) -> dict:
    return {
        "in_domain": bool(ad.in_domain),
        "nearest_neighbor_tanimoto": float(ad.nearest_neighbor_tanimoto),
        "threshold": float(ad.threshold),
    }


# ---------------------------------------------------- health / fingerprint


def _sha256(path: Path) -> str:
    h = _hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def vina_version() -> str | None:
    """Return the runtime Vina version string, or None if unavailable."""
    try:
        out = _subprocess.check_output(
            ["vina", "--version"], stderr=_subprocess.STDOUT, timeout=5
        )
        return out.decode().strip().splitlines()[0]
    except (FileNotFoundError, _subprocess.SubprocessError):
        return None


def git_sha(repo_path: Path | str = ".") -> str | None:
    try:
        out = _subprocess.check_output(
            ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
            stderr=_subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode().strip()
    except (FileNotFoundError, _subprocess.SubprocessError):
        return None


def health_payload(
    *,
    model_path: Path | str,
    model_name: str,
    dataset_version: str | None = None,
    started_at_utc: str | None = None,
    extra: dict[str, _Any] | None = None,
) -> dict:
    model_path = Path(model_path)
    payload = {
        "status": "ok" if model_path.exists() else "model_missing",
        "model_name": model_name,
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path) if model_path.exists() else None,
        "dataset_version": dataset_version,
        "vina_version": vina_version(),
        "git_sha": git_sha(),
        "python_version": _platform.python_version(),
        "platform": _platform.platform(),
        "started_at_utc": started_at_utc
        or _datetime.now(_timezone.utc).isoformat(),
        "pid": os.getpid(),
    }
    if extra:
        payload.update(extra)
    return payload


def write_health_snapshot(payload: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json2.dumps(payload, indent=2, sort_keys=True))
    return path


# =====================================================================
# Stage 12 / Phase 5 — ModernBiasPredictor
#
# Loads the new sklearn-Pipeline-based artifacts produced by the rewritten
# Stage 10 trainer (`data/processed/ml_models/<model>_final[_calibrated].joblib`).
# Builds the per-prediction feature row using the SAME helper the trainer
# uses (`cancerag.ml.preprocessing.get_X_y_groups`) so structural feature
# parity is guaranteed by construction (see tests/unit/test_feature_parity.py).
# =====================================================================


import joblib as _joblib  # noqa: E402


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


class ModernBiasPredictor:
    """Loads a sklearn Pipeline (preprocessing + estimator) saved by the
    rewritten Stage 10 trainer.

    Resolves the chosen model from ``selection_decision.json`` and prefers
    the calibrated artifact when present. Optionally loads the stacking
    ensemble (LogReg meta-learner over [LightGBM, XGBoost, ChemBERTa-LR]
    OOF probabilities).

    Builds per-prediction feature rows by reusing
    :func:`cancerag.ml.preprocessing.get_X_y_groups` so the feature schema
    matches training by construction. Pre-computed Vina / IFP / 3D pose
    features are looked up by ``(inchikey, receptor_uniprot)`` from the
    cached CSV/parquet artifacts; if no row exists for the pair, columns
    are NaN-padded and the in-pipeline imputer fills them.
    """

    def __init__(
        self,
        selection_decision_path: Path | str | None = None,
        *,
        repo_root: Path | str | None = None,
        use_stacking: bool = False,
        ad_threshold: float = 0.4,
    ):
        self.repo_root = Path(repo_root) if repo_root else _default_repo_root()
        if selection_decision_path is None:
            selection_decision_path = (
                self.repo_root
                / "data" / "processed" / "ml_models" / "selection_decision.json"
            )
        self.selection_decision_path = Path(selection_decision_path)
        self.use_stacking = bool(use_stacking)
        self.ad_threshold = float(ad_threshold)

        self.model = None              # primary sklearn Pipeline
        self.model_path: Path | None = None
        self.model_sha256: str | None = None
        self.model_name: str | None = None
        self.label_encoder = None
        self.feature_columns: list[str] = []
        self._train_X: pd.DataFrame | None = None
        self._train_df: pd.DataFrame | None = None
        self.applicability: ApplicabilityChecker | None = None
        self._explainer = None
        self._explainer_kind: str | None = None  # "tree" or "kernel"
        # Stacking ensemble (optional)
        self.stacking_meta = None
        self.stacking_bases: dict = {}

        self._load()

    # ------------------------------------------------------------- loading

    def _load(self) -> None:
        if not self.selection_decision_path.exists():
            raise FileNotFoundError(
                f"selection_decision.json missing: {self.selection_decision_path}"
            )
        decision = _json2.loads(self.selection_decision_path.read_text())
        winner = decision.get("chosen")
        if not isinstance(winner, str) or not winner:
            raise ValueError(
                f"selection_decision.json missing 'chosen': {decision!r}"
            )
        self.model_name = winner
        models_dir = self.selection_decision_path.parent
        cal_path = models_dir / f"{winner}_final_calibrated.joblib"
        final_path = models_dir / f"{winner}_final.joblib"
        if cal_path.exists():
            self.model_path = cal_path
        elif final_path.exists():
            self.model_path = final_path
        else:
            raise FileNotFoundError(
                f"No model artifact for winner={winner!r} in {models_dir}"
            )
        logger.info("ModernBiasPredictor loading model: %s", self.model_path)
        self.model = _joblib.load(self.model_path)
        self.model_sha256 = _sha256(self.model_path)

        # Label encoder
        le_path = (
            self.repo_root
            / "data" / "processed" / "ml_preprocessed" / "label_encoder.joblib"
        )
        if le_path.exists():
            self.label_encoder = _joblib.load(le_path)
        else:
            logger.warning("label_encoder.joblib not found at %s", le_path)

        # Training dataset (for AD + feature schema)
        ds_path = (
            self.repo_root
            / "data" / "processed" / "ml_ready_dataset.parquet"
        )
        if ds_path.exists():
            from cancerag.ml.preprocessing import get_X_y_groups
            self._train_df = pd.read_parquet(ds_path)
            X, _y, _sw, le, _cols = get_X_y_groups(
                self._train_df, label_encoder=self.label_encoder
            )
            if self.label_encoder is None:
                self.label_encoder = le
            self._train_X = X
            self.feature_columns = list(X.columns)
            train_smiles = (
                self._train_df["canonical_smiles"].dropna().astype(str).tolist()
            )
            try:
                self.applicability = ApplicabilityChecker(
                    train_smiles, threshold=self.ad_threshold
                )
            except ValueError:
                self.applicability = None
        else:
            logger.warning("ml_ready_dataset.parquet not found at %s", ds_path)

        # Optional: load stacking ensemble
        if self.use_stacking:
            self._load_stacking()

        # Build SHAP explainer (cached for reuse)
        self._build_explainer()

    def _load_stacking(self) -> None:
        adv = self.repo_root / "data" / "processed" / "ml_models" / "advanced"
        meta_path = adv / "stacking_meta_learner.joblib"
        if not meta_path.exists():
            logger.warning("Stacking meta-learner missing at %s", meta_path)
            return
        self.stacking_meta = _joblib.load(meta_path)
        # Base models — try the calibrated/final variants in standard locations
        base_candidates = {
            "lightgbm": [
                adv / "lightgbm_tuned_calibrated.joblib",
                self.repo_root / "data/processed/ml_models/lightgbm_final_calibrated.joblib",
                self.repo_root / "data/processed/ml_models/lightgbm_final.joblib",
            ],
            "xgboost": [
                self.repo_root / "data/processed/ml_models/xgboost_final.joblib",
            ],
            "chemberta_lr": [
                self.repo_root
                / "data/processed/ml_models/baselines/chemberta/chemberta_logreg_calibrated.joblib",
            ],
        }
        for name, paths in base_candidates.items():
            for p in paths:
                if p.exists():
                    try:
                        self.stacking_bases[name] = _joblib.load(p)
                        logger.info("Loaded stacking base %s from %s", name, p)
                        break
                    except Exception as exc:
                        logger.warning("Failed to load %s: %s", p, exc)

    def _build_explainer(self) -> None:
        if self.model is None:
            return
        try:
            import shap  # local import to keep import time low
        except ImportError:
            logger.warning("shap not available — SHAP explanations disabled")
            return
        # The model is a Pipeline (preprocessing + estimator) or a calibrated
        # wrapper. SHAP's TreeExplainer needs the raw tree estimator AND
        # features in its post-preprocessing space, so we also capture the
        # preprocessing prefix to transform inputs at explain-time.
        est = self._extract_tree_estimator(self.model)
        if est is not None:
            try:
                self._explainer = shap.TreeExplainer(est)
                self._explainer_kind = "tree"
                self._preprocessing_prefix = self._extract_preprocessing_prefix(
                    self.model
                )
                logger.info("SHAP TreeExplainer initialised on %s", type(est).__name__)
                return
            except Exception as exc:
                logger.warning("TreeExplainer failed (%s); falling back", exc)
        # Fallback: KernelExplainer over predict_proba on a small training
        # background sample. Used for the LR meta-learner and any non-tree
        # models.
        try:
            background = self._sample_background(50)
            if background is None:
                logger.warning("No background data — SHAP disabled")
                return
            self._explainer = shap.KernelExplainer(
                self.model.predict_proba, background
            )
            self._explainer_kind = "kernel"
            logger.info("SHAP KernelExplainer initialised")
        except Exception as exc:
            logger.warning("KernelExplainer init failed: %s", exc)
            self._explainer = None
            self._explainer_kind = None

    def _extract_preprocessing_prefix(self, obj):
        """Return a callable that transforms a raw-feature DataFrame through
        the preprocessing prefix of the saved Pipeline (everything except
        the final estimator step). Returns identity if no Pipeline wraps."""
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            return lambda X: X
        if isinstance(obj, Pipeline) and len(obj.steps) > 1:
            prefix = Pipeline(obj.steps[:-1])
            return prefix.transform
        # CalibratedClassifierCV may wrap a Pipeline
        for attr in ("estimator", "base_estimator"):
            inner = getattr(obj, attr, None)
            if inner is not None and inner is not obj:
                fn = self._extract_preprocessing_prefix(inner)
                if fn is not None:
                    return fn
        if hasattr(obj, "calibrated_classifiers_") and obj.calibrated_classifiers_:
            inner = obj.calibrated_classifiers_[0]
            for attr in ("estimator", "base_estimator"):
                cand = getattr(inner, attr, None)
                if cand is not None:
                    fn = self._extract_preprocessing_prefix(cand)
                    if fn is not None:
                        return fn
        return lambda X: X

    def _extract_tree_estimator(self, obj):
        """Walk through Pipeline / CalibratedClassifierCV wrappers and return
        the underlying tree estimator if present, else None."""
        try:
            from sklearn.pipeline import Pipeline
        except ImportError:
            Pipeline = None
        # Pipeline: take the last step
        if Pipeline is not None and isinstance(obj, Pipeline):
            return self._extract_tree_estimator(obj.steps[-1][1])
        # CalibratedClassifierCV (sklearn >=0.24): use one of the underlying
        # calibrated_classifiers_[i].estimator (tree) — but TreeExplainer
        # works fine on the calibrated booster directly via its internal
        # estimator attribute. We try common attrs:
        for attr in ("estimator", "base_estimator"):
            if hasattr(obj, attr):
                inner = getattr(obj, attr)
                if inner is not None and inner is not obj:
                    sub = self._extract_tree_estimator(inner)
                    if sub is not None:
                        return sub
        # CalibratedClassifierCV stores fitted base learners in
        # calibrated_classifiers_
        if hasattr(obj, "calibrated_classifiers_") and obj.calibrated_classifiers_:
            inner = obj.calibrated_classifiers_[0]
            for attr in ("estimator", "base_estimator"):
                if hasattr(inner, attr):
                    cand = getattr(inner, attr)
                    sub = self._extract_tree_estimator(cand)
                    if sub is not None:
                        return sub
        # Tree-likes
        cls_name = type(obj).__name__
        tree_likes = {
            "LGBMClassifier", "XGBClassifier", "RandomForestClassifier",
            "GradientBoostingClassifier", "ExtraTreesClassifier",
            "HistGradientBoostingClassifier",
        }
        if cls_name in tree_likes:
            return obj
        return None

    def _sample_background(self, n: int) -> pd.DataFrame | None:
        if self._train_X is None or self._train_X.empty:
            return None
        n = min(n, len(self._train_X))
        return self._train_X.sample(n=n, random_state=0).reset_index(drop=True)

    # ------------------------------------------------------------- features

    def build_input_row(
        self, smiles: str, receptor_uniprot: str
    ) -> pd.DataFrame:
        """Construct a single-row DataFrame matching the training feature schema.

        Strategy: look the (inchikey, receptor_uniprot) pair up in the
        ml_ready_dataset; if it exists, use its feature row. Otherwise build a
        NaN-padded row with the canonical SMILES + receptor and let the
        in-pipeline imputer fill in the missing columns.
        """
        if self._train_X is None or self._train_df is None:
            raise RuntimeError(
                "Training dataset not loaded; cannot build input row."
            )
        from rdkit import Chem
        from rdkit.Chem.inchi import InchiToInchiKey, MolToInchi
        # Resolve InChIKey from the SMILES so we can join against pre-computed
        # docking / IFP features.
        ikey = None
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                inchi = MolToInchi(mol)
                ikey = InchiToInchiKey(inchi) if inchi else None
        except Exception:
            ikey = None
        # Try by (inchikey, receptor_uniprot)
        if ikey is not None and "inchikey" in self._train_df.columns:
            mask = (
                (self._train_df["inchikey"] == ikey)
                & (self._train_df["receptor_uniprot"] == receptor_uniprot)
            )
            if mask.any():
                idx = mask.idxmax()
                pos = self._train_df.index.get_loc(idx)
                return self._train_X.iloc[[pos]].copy().reset_index(drop=True)
        # Fallback: NaN-padded row matching schema
        row = pd.DataFrame(
            [[np.nan] * len(self.feature_columns)],
            columns=self.feature_columns,
        )
        return row

    # ------------------------------------------------------------- predict

    def predict(
        self,
        smiles: str,
        receptor_uniprot: str,
        *,
        log_audit: bool = True,
    ) -> dict:
        """Run a single-pair prediction.

        Returns a dict with: predicted_class, probabilities, applicability,
        confidence, top_shap, model_name, model_sha256.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")
        X_row = self.build_input_row(smiles, receptor_uniprot)
        if self.use_stacking and self.stacking_meta is not None and self.stacking_bases:
            proba = self._predict_stacking(X_row)
        else:
            proba = self.model.predict_proba(X_row)[0]
        classes = self._class_names()
        pred_idx = int(np.argmax(proba))
        predicted_class = classes[pred_idx] if pred_idx < len(classes) else f"Class_{pred_idx}"
        probabilities = {
            classes[i] if i < len(classes) else f"Class_{i}": float(p)
            for i, p in enumerate(proba)
        }
        # Applicability
        if self.applicability is not None:
            ad = self.applicability.check(smiles)
        else:
            ad = ADResult(in_domain=True, nearest_neighbor_tanimoto=0.0,
                          threshold=self.ad_threshold)
        conf = confidence_label(ad, max_class_proba=float(np.max(proba)))
        # SHAP top-5
        top_shap = self._top_shap(X_row, pred_idx)
        result = {
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "applicability": ad_to_dict(ad),
            "confidence": conf,
            "top_shap": top_shap,
            "model_name": self.model_name,
            "model_sha256": self.model_sha256,
        }
        if log_audit:
            self._append_audit(smiles, receptor_uniprot, result)
        return result

    def _predict_stacking(self, X_row: pd.DataFrame) -> np.ndarray:
        """Concat base model OOF probas → meta.predict_proba.

        For ChemBERTa we don't have a pretrained text→embedding pipeline at
        inference time bundled here; if its model expects raw embeddings we
        fall back to uniform probas. The intention of `use_stacking=True`
        is to load the artifacts and demonstrate the path; production use
        would pre-compute the ChemBERTa embedding for the SMILES.
        """
        proba_chunks: list[np.ndarray] = []
        for name in ("chemberta_lr", "lightgbm", "xgboost"):
            base = self.stacking_bases.get(name)
            if base is None:
                # Uniform fallback if a base is missing
                n_classes = len(self._class_names())
                proba_chunks.append(np.full((1, n_classes), 1.0 / n_classes))
                continue
            try:
                p = base.predict_proba(X_row)
            except Exception:
                n_classes = len(self._class_names())
                p = np.full((1, n_classes), 1.0 / n_classes)
            proba_chunks.append(p)
        stacked = np.concatenate(proba_chunks, axis=1)
        return self.stacking_meta.predict_proba(stacked)[0]

    def _class_names(self) -> list[str]:
        if self.label_encoder is not None and hasattr(self.label_encoder, "classes_"):
            return [str(c) for c in self.label_encoder.classes_]
        # Try the model
        if hasattr(self.model, "classes_"):
            return [str(c) for c in self.model.classes_]
        return []

    def _top_shap(
        self, X_row: pd.DataFrame, pred_idx: int, k: int = 5
    ) -> list[tuple[str, float]]:
        if self._explainer is None:
            return []
        try:
            if self._explainer_kind == "tree":
                # Transform through the preprocessing prefix so feature
                # space matches what the tree estimator was trained on.
                transform = getattr(
                    self, "_preprocessing_prefix", lambda X: X
                )
                X_for_explain = transform(X_row)
                sv = self._explainer.shap_values(X_for_explain)
                cols = (
                    list(X_for_explain.columns)
                    if hasattr(X_for_explain, "columns")
                    else [f"f{i}" for i in range(np.asarray(X_for_explain).shape[1])]
                )
            else:
                # KernelExplainer is slow; use small nsamples
                sv = self._explainer.shap_values(X_row, nsamples=50, silent=True)
                cols = list(X_row.columns)
            # sv may be: list-of-arrays (one per class), single 2D array,
            # or 3D (n_rows, n_features, n_classes)
            arr = self._select_shap_for_class(sv, pred_idx)
            if arr is None:
                return []
            order = np.argsort(np.abs(arr))[::-1][:k]
            return [(str(cols[int(i)]), float(arr[int(i)])) for i in order]
        except Exception as exc:
            logger.warning("SHAP computation failed: %s", exc)
            return []

    @staticmethod
    def _select_shap_for_class(sv, pred_idx: int):
        if isinstance(sv, list):
            # multi-class list: index by class then take row 0
            if pred_idx < len(sv):
                return np.asarray(sv[pred_idx])[0]
            return np.asarray(sv[0])[0]
        arr = np.asarray(sv)
        if arr.ndim == 3:  # (n_rows, n_features, n_classes)
            cls = pred_idx if pred_idx < arr.shape[2] else 0
            return arr[0, :, cls]
        if arr.ndim == 2:  # (n_rows, n_features)
            return arr[0]
        if arr.ndim == 1:
            return arr
        return None

    # ------------------------------------------------------------- audit

    def _append_audit(
        self, smiles: str, receptor_uniprot: str, result: dict
    ) -> None:
        audit_path = (
            self.repo_root / "data" / "processed" / "inference_audit.jsonl"
        )
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        smiles_sha = _hashlib.sha256(
            (smiles or "").encode("utf-8")
        ).hexdigest()
        entry = {
            "ts_utc": _datetime.now(_timezone.utc).isoformat(),
            "smiles_sha256": smiles_sha,
            "receptor_uniprot": receptor_uniprot,
            "predicted_class": result["predicted_class"],
            "probabilities": result["probabilities"],
            "top_shap": result["top_shap"],
            "model_name": result.get("model_name"),
            "model_sha256": result.get("model_sha256"),
        }
        with audit_path.open("a") as f:
            f.write(_json2.dumps(entry) + "\n")
