from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol
import pandas as pd


# -------------------- Stage Result --------------------
@dataclass
class StageResult:
    stage_name: str
    status: str  # "ok" | "error"
    message: str
    in_shape: tuple[int, int]
    out_shape: tuple[int, int]
    preview_head: pd.DataFrame
    stats: Dict[str, Any] = field(default_factory=dict)

    # --- Schema diff fields ---
    added_cols: List[str] = field(default_factory=list)
    removed_cols: List[str] = field(default_factory=list)
    kept_cols: List[str] = field(default_factory=list)
    modified_cols: List[str] = field(default_factory=list)


# -------------------- Data Packet --------------------
@dataclass
class DataPacket:
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)
    history: List[StageResult] = field(default_factory=list)


# -------------------- Filter Interface --------------------
class Filter(Protocol):
    name: str

    def run(self, packet: DataPacket) -> DataPacket:
        ...


# -------------------- Pipeline Runner --------------------
class PipelineRunner:
    def __init__(self, filters: List[Filter]):
        self.filters = filters

    def run(self, packet: DataPacket, preview_rows: int = 8) -> DataPacket:
        for f in self.filters:
            in_shape = packet.df.shape
            before_cols = list(packet.df.columns)

            try:
                packet = f.run(packet)

                out_shape = packet.df.shape
                after_cols = list(packet.df.columns)

                before_set = set(before_cols)
                after_set = set(after_cols)

                added = sorted(after_set - before_set)
                removed = sorted(before_set - after_set)
                kept = sorted(before_set & after_set)

                # Columns modified in-place (values changed, schema same)
                modified = packet.meta.get(f"modified::{f.name}", [])
                if not isinstance(modified, list):
                    modified = []
                modified = [c for c in modified if c in after_set]

                packet.history.append(
                    StageResult(
                        stage_name=f.name,
                        status="ok",
                        message="Completed successfully.",
                        in_shape=in_shape,
                        out_shape=out_shape,
                        preview_head=packet.df.head(preview_rows).copy(),
                        stats=packet.meta.get(f"stats::{f.name}", {}),
                        added_cols=added,
                        removed_cols=removed,
                        kept_cols=kept,
                        modified_cols=modified,
                    )
                )

            except Exception as e:
                # Stop pipeline on error
                packet.history.append(
                    StageResult(
                        stage_name=f.name,
                        status="error",
                        message=str(e),
                        in_shape=in_shape,
                        out_shape=in_shape,
                        preview_head=packet.df.head(preview_rows).copy(),
                        stats={},
                        added_cols=[],
                        removed_cols=[],
                        kept_cols=before_cols,
                        modified_cols=[],
                    )
                )
                break

        return packet
