from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class ValidationMessage:
    level: str   # "error" | "warning"
    text: str


def validate_pipeline(step_keys: List[str]) -> List[ValidationMessage]:
    """
    Validates pipeline structure (ordering / logical constraints).
    This is pattern-level validation (not data-level).
    """
    msgs: List[ValidationMessage] = []

    # Helper: index lookup
    idx = {k: i for i, k in enumerate(step_keys)}

    def before(a: str, b: str) -> bool:
        return a in idx and b in idx and idx[a] < idx[b]

    def after(a: str, b: str) -> bool:
        return a in idx and b in idx and idx[a] > idx[b]

    # ---- Rules ----

    # PCA should happen after scale (recommended), and after encoding (if categoricals exist)
    if "pca" in idx:
        if "scale" in idx and not after("pca", "scale"):
            msgs.append(ValidationMessage(
                "error",
                "PCA should run AFTER Scaling. Move 'Scale' before 'PCA'."
            ))
        if "encode" in idx and not after("pca", "encode"):
            msgs.append(ValidationMessage(
                "warning",
                "PCA is usually applied AFTER Encoding when categoricals exist. Consider moving 'Encode' before 'PCA'."
            ))

        # If PCA exists without scaling, warn (not always fatal, but usually wrong)
        if "scale" not in idx:
            msgs.append(ValidationMessage(
                "warning",
                "PCA without scaling can be biased by feature magnitudes. Consider adding 'Scale' before 'PCA'."
            ))

    # Scaling before encoding? Usually scaling should be after encoding (since one-hot creates numeric columns too)
    if "scale" in idx and "encode" in idx and before("scale", "encode"):
        msgs.append(ValidationMessage(
            "warning",
            "Scaling is typically done AFTER Encoding (since encoding changes the feature space). Consider swapping 'Encode' and 'Scale'."
        ))

    # Impute should generally be early
    if "impute" in idx:
        for later in ["encode", "scale", "pca"]:
            if later in idx and after("impute", later):
                msgs.append(ValidationMessage(
                    "warning",
                    "Imputation is usually done BEFORE other preprocessing steps. Consider moving 'Impute' earlier."
                ))
                break

    # Empty pipeline
    if len(step_keys) == 0:
        msgs.append(ValidationMessage("error", "Pipeline has no filters. Please select at least one filter."))

    return msgs
