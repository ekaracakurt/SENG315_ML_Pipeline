from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd


@dataclass
class ValidationMessage:
    level: str   # "error" | "warning"
    text: str


def _estimate_feature_count_at_pca(
    df: pd.DataFrame,
    step_keys: List[str],
    params: Dict[str, Dict[str, Any]],
) -> int:
    """
    Estimate number of numeric features available when PCA runs,
    taking into account whether encoding happens before PCA.
    """
    # Start with raw numeric columns
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # If encode is not present OR it happens after PCA, categoricals won't be numeric for PCA stage
    if "encode" not in step_keys or "pca" not in step_keys:
        return len(num_cols)

    idx = {k: i for i, k in enumerate(step_keys)}
    if idx["encode"] > idx["pca"]:
        # encoding occurs after PCA -> PCA sees only current numeric cols
        return len(num_cols)

    # Encoding occurs before PCA -> estimate one-hot expanded count
    drop = params.get("encode", {}).get("drop", None)  # None or "first"
    # scikit's OneHotEncoder creates one feature per category (minus 1 if drop="first")
    onehot_features = 0
    for c in cat_cols:
        # nunique(dropna=False) counts NaN as a category too, which matches typical encoder behavior when not imputed.
        k = int(df[c].nunique(dropna=False))
        if k <= 0:
            continue
        if drop == "first" and k > 1:
            k -= 1
        onehot_features += k

    # After encoding, dataset typically becomes: numeric + onehot (our encoder filter also does that)
    return len(num_cols) + onehot_features


def validate_pipeline_with_data(
    df: pd.DataFrame,
    step_keys: List[str],
    params: Dict[str, Dict[str, Any]] | None = None,
) -> List[ValidationMessage]:
    """
    Data-aware validation that also estimates schema at PCA stage
    based on pipeline order and encoder settings.
    """
    params = params or {}
    msgs: List[ValidationMessage] = []

    if df is None or df.empty:
        msgs.append(ValidationMessage("error", "Dataset is empty or not loaded."))
        return msgs

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    missing_total = int(df.isna().sum().sum())

    idx = {k: i for i, k in enumerate(step_keys)}

    def after(a: str, b: str) -> bool:
        return a in idx and b in idx and idx[a] > idx[b]

    # Missing values exist but impute not selected
    if missing_total > 0 and "impute" not in step_keys:
        msgs.append(ValidationMessage(
            "warning",
            f"Dataset contains {missing_total} missing values, but 'Impute' is not enabled."
        ))

    # Encode selected but no categoricals
    if "encode" in step_keys and len(cat_cols) == 0:
        msgs.append(ValidationMessage("warning", "Encode is enabled, but no categorical columns were detected."))

    # PCA sanity checks
    if "pca" in step_keys:
        # If categoricals exist and encode is missing or after PCA -> PCA will not be able to handle them
        if len(cat_cols) > 0:
            if "encode" not in step_keys:
                msgs.append(ValidationMessage(
                    "error",
                    "PCA is enabled but dataset contains categorical columns and 'Encode' is disabled. PCA needs numeric features."
                ))
            elif not after("pca", "encode"):
                msgs.append(ValidationMessage(
                    "error",
                    "PCA must run AFTER Encoding when categorical columns exist."
                ))

        # Estimate feature count at PCA stage (THIS is the important fix)
        n_samples = len(df)
    est_features = _estimate_feature_count_at_pca(df, step_keys, params)
    n_components = int(params.get("pca", {}).get("n_components", 5))

    max_allowed = min(n_samples, est_features)

    if max_allowed <= 0:
        msgs.append(
            ValidationMessage(
                "error",
                "PCA cannot run because there will be no valid features or samples at PCA stage."
            )
        )
    elif n_components > max_allowed:
        msgs.append(
            ValidationMessage(
                "error",
                f"PCA n_components ({n_components}) exceeds the maximum allowed value "
                f"min(n_samples={n_samples}, n_features={est_features}) = {max_allowed}. "
                "Reduce n_components or provide more data."
            )
        )


        

    return msgs
