from dataclasses import dataclass
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pipeline.core import DataPacket


@dataclass
class CategoricalEncoderFilter:
    name: str = "One-Hot Encode Categoricals"
    drop: str | None = None              # None or "first"
    handle_unknown: str = "ignore"

    def run(self, packet: DataPacket) -> DataPacket:
        df = packet.df.copy()

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        if not cat_cols:
            packet.meta[f"stats::{self.name}"] = {"note": "No categorical columns found."}
            packet.df = df
            packet.meta[f"modified::{self.name}"] = []
            return packet

        enc = OneHotEncoder(drop=self.drop, handle_unknown=self.handle_unknown, sparse_output=False)
        encoded = enc.fit_transform(df[cat_cols])

        encoded_cols = enc.get_feature_names_out(cat_cols).tolist()
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)

        df_out = pd.concat([df[num_cols], encoded_df], axis=1)

        packet.df = df_out
        packet.meta[f"stats::{self.name}"] = {
            "cat_cols_removed": cat_cols,
            "num_cols_kept": num_cols,
            "new_columns": len(encoded_cols),
            "drop": self.drop,
        }

        # Encoding changes schema (handled by schema diff), values are new columns, so no "modified"
        packet.meta[f"modified::{self.name}"] = []
        return packet
