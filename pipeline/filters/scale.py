from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pipeline.core import DataPacket


@dataclass
class ScalerFilter:
    name: str = "Scale Numeric Features"
    method: str = "standard"  # "standard" or "minmax"

    def run(self, packet: DataPacket) -> DataPacket:
        df = packet.df.copy()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if not num_cols:
            packet.meta[f"stats::{self.name}"] = {"note": "No numeric columns found."}
            packet.df = df
            packet.meta[f"modified::{self.name}"] = []
            return packet

        scaler = StandardScaler() if self.method == "standard" else MinMaxScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])

        packet.df = df
        packet.meta[f"stats::{self.name}"] = {"num_cols": num_cols, "method": self.method}

        # Scaling modifies values in-place (schema unchanged)
        packet.meta[f"modified::{self.name}"] = num_cols
        return packet
