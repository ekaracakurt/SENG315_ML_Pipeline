from dataclasses import dataclass
import pandas as pd
from sklearn.impute import SimpleImputer
from pipeline.core import DataPacket


@dataclass
class MissingValueImputerFilter:
    name: str = "Impute Missing Values"
    strategy_num: str = "median"   # median/mean
    strategy_cat: str = "most_frequent"

    def run(self, packet: DataPacket) -> DataPacket:
        df = packet.df.copy()
        before_missing = int(df.isna().sum().sum())

        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = [c for c in df.columns if c not in num_cols]

        if num_cols:
            imp_num = SimpleImputer(strategy=self.strategy_num)
            df[num_cols] = imp_num.fit_transform(df[num_cols])

        if cat_cols:
            imp_cat = SimpleImputer(strategy=self.strategy_cat)
            df[cat_cols] = imp_cat.fit_transform(df[cat_cols])

        after_missing = int(df.isna().sum().sum())

        packet.df = df
        packet.meta[f"stats::{self.name}"] = {
            "missing_before": before_missing,
            "missing_after": after_missing,
            "num_cols": num_cols,
            "cat_cols": cat_cols,
            "strategies": {"num": self.strategy_num, "cat": self.strategy_cat},
        }

        # Values may change in these columns (schema unchanged)
        packet.meta[f"modified::{self.name}"] = num_cols + cat_cols
        return packet
