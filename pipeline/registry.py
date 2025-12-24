from pipeline.filters.impute import MissingValueImputerFilter
from pipeline.filters.encode import CategoricalEncoderFilter
from pipeline.filters.scale import ScalerFilter
from pipeline.filters.pca import PCAFeatureExtractionFilter

FILTER_CATALOG = {
    "impute": ("Impute Missing Values", MissingValueImputerFilter),
    "encode": ("One-Hot Encode Categoricals", CategoricalEncoderFilter),
    "scale": ("Scale Numeric Features", ScalerFilter),
    "pca": ("PCA Feature Extraction", PCAFeatureExtractionFilter),
}

def build_filter(key: str, params: dict):
    if key not in FILTER_CATALOG:
        raise KeyError(f"Unknown filter key: {key}")
    _, cls = FILTER_CATALOG[key]
    return cls(**params)
