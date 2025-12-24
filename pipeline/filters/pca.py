from dataclasses import dataclass
import pandas as pd
from sklearn.decomposition import PCA
from pipeline.core import DataPacket


@dataclass
class PCAFeatureExtractionFilter:
    name: str = "PCA Feature Extraction"
    n_components: int = 5

    def run(self, packet: DataPacket) -> DataPacket:
        df = packet.df.copy()

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        if not num_cols:
            raise ValueError("PCA requires numeric features, but none were found.")

        X = df[num_cols].to_numpy()
        n = min(self.n_components, X.shape[1], X.shape[0])  # also cap by n_samples
        if n < 1:
            raise ValueError("n_components must be >= 1 and <= min(n_samples, n_features).")

        pca = PCA(n_components=n)
        Z = pca.fit_transform(X)

        out_cols = [f"PC{i+1}" for i in range(Z.shape[1])]
        out_df = pd.DataFrame(Z, columns=out_cols, index=df.index)

        packet.df = out_df
        packet.meta[f"stats::{self.name}"] = {
            "input_features": len(num_cols),
            "output_features": len(out_cols),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "n_components_used": n,
        }

        # PCA replaces columns (schema diff will show it)
        packet.meta[f"modified::{self.name}"] = []
        return packet
