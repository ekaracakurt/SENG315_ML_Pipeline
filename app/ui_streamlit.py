import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import pandas as pd
import streamlit as st

from pipeline.validate_data import validate_pipeline_with_data
from pipeline.validate import validate_pipeline
from pipeline.core import DataPacket, PipelineRunner
from pipeline.registry import FILTER_CATALOG, build_filter

st.set_page_config(page_title="Pipe-and-Filter ML Preprocessing", layout="wide")
st.title("Pipe-and-Filter: ML Preprocessing Pipeline")

st.sidebar.header("1) Load Dataset")
uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded is not None:
    st.session_state.df = pd.read_csv(uploaded)

df = st.session_state.df
if df is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# -------------------- Dataset Preview --------------------
st.subheader("Dataset Preview")
st.dataframe(df.head(min(20, len(df))), use_container_width=True)
st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

st.subheader("Dataset Summary")
num_cols = df.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in df.columns if c not in num_cols]
missing_total = int(df.isna().sum().sum())
st.write(f"Numeric columns ({len(num_cols)}): {num_cols}")
st.write(f"Categorical columns ({len(cat_cols)}): {cat_cols}")
st.write(f"Total missing values: {missing_total}")

# -------------------- Pipeline Builder --------------------
st.sidebar.header("2) Build Pipeline")

if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = ["impute", "encode", "scale", "pca"]

available_keys = list(FILTER_CATALOG.keys())

selected = st.sidebar.multiselect(
    "Enable filters",
    options=available_keys,
    default=st.session_state.pipeline_steps,
)

# Preserve prior order + append new selections
ordered = [k for k in st.session_state.pipeline_steps if k in selected] + [
    k for k in selected if k not in st.session_state.pipeline_steps
]
st.session_state.pipeline_steps = ordered

st.sidebar.markdown("**Order filters**")
for i, key in enumerate(st.session_state.pipeline_steps):
    col1, col2, col3 = st.sidebar.columns([6, 1, 1])
    col1.write(FILTER_CATALOG[key][0])

    if col2.button("↑", key=f"up_{key}") and i > 0:
        steps = st.session_state.pipeline_steps
        steps[i - 1], steps[i] = steps[i], steps[i - 1]
        st.session_state.pipeline_steps = steps
        st.rerun()

    if col3.button("↓", key=f"down_{key}") and i < len(st.session_state.pipeline_steps) - 1:
        steps = st.session_state.pipeline_steps
        steps[i + 1], steps[i] = steps[i], steps[i + 1]
        st.session_state.pipeline_steps = steps
        st.rerun()

# -------------------- Filter Configuration --------------------
st.sidebar.header("3) Configure Filters")

params = {}
for key in st.session_state.pipeline_steps:
    label, _ = FILTER_CATALOG[key]
    with st.sidebar.expander(label, expanded=False):
        if key == "impute":
            params[key] = {
                "strategy_num": st.selectbox("Numeric strategy", ["median", "mean"], index=0, key="impute_num"),
                "strategy_cat": st.selectbox("Categorical strategy", ["most_frequent"], index=0, key="impute_cat"),
            }
        elif key == "encode":
            drop = st.selectbox("Drop first?", ["None", "first"], index=0, key="encode_drop")
            params[key] = {"drop": None if drop == "None" else "first"}
        elif key == "scale":
            params[key] = {"method": st.selectbox("Method", ["standard", "minmax"], index=0, key="scale_method")}
        elif key == "pca":
            params[key] = {
                "n_components": st.number_input(
                    "n_components", min_value=1, max_value=200, value=5, step=1, key="pca_n"
                )
            }

# -------------------- Validation (Structure + Data) --------------------
structure_msgs = validate_pipeline(st.session_state.pipeline_steps)
data_msgs = validate_pipeline_with_data(df, st.session_state.pipeline_steps, params)
all_msgs = structure_msgs + data_msgs

errors = [m for m in all_msgs if m.level == "error"]
warnings = [m for m in all_msgs if m.level == "warning"]

if errors:
    st.sidebar.error("Pipeline errors (must fix before running):")
    for m in errors:
        st.sidebar.write(f"• {m.text}")

if warnings:
    st.sidebar.warning("Pipeline warnings:")
    for m in warnings:
        st.sidebar.write(f"• {m.text}")

# -------------------- Run + Export --------------------
st.sidebar.header("4) Run")
run_btn = st.sidebar.button("Run Pipeline", type="primary", disabled=len(errors) > 0)

colA, colB = st.columns([1, 1])

with colA:
    st.subheader("Pipeline Steps (execution order)")
    for idx, key in enumerate(st.session_state.pipeline_steps, start=1):
        st.write(f"{idx}. {FILTER_CATALOG[key][0]}")

with colB:
    st.subheader("Export Pipeline Config")
    pipeline_config = {"steps": [{"key": k, "params": params.get(k, {})} for k in st.session_state.pipeline_steps]}
    st.download_button(
        "Download pipeline.json",
        data=json.dumps(pipeline_config, indent=2),
        file_name="pipeline.json",
        mime="application/json",
    )

# -------------------- Execute Pipeline --------------------
if run_btn:
    filters = [build_filter(step["key"], step["params"]) for step in pipeline_config["steps"]]
    runner = PipelineRunner(filters)
    packet = DataPacket(df=df.copy())
    out = runner.run(packet, preview_rows=min(20, len(df)))

    st.subheader("Stage Results")
    for res in out.history:
        with st.expander(f"{res.stage_name} — {res.status.upper()}"):
            st.write(res.message)
            st.write(f"Shape: {res.in_shape} → {res.out_shape}")

            # --- Schema diff visualization ---
            st.markdown("### Schema Diff")
            c1, c2, c3 = st.columns(3)
            c1.metric("Added columns", len(res.added_cols))
            c2.metric("Removed columns", len(res.removed_cols))
            c3.metric("Kept columns", len(res.kept_cols))

            if res.added_cols:
                st.markdown("**Added:**")
                st.code(", ".join(res.added_cols), language="text")

            if res.removed_cols:
                st.markdown("**Removed:**")
                st.code(", ".join(res.removed_cols), language="text")

            if res.modified_cols:
                st.markdown("**Modified (values changed, same columns):**")
                st.code(", ".join(res.modified_cols), language="text")

            # Stats + preview
            if res.stats:
                st.markdown("### Stage Stats")
                st.json(res.stats)

            st.markdown("### Output Preview")
            st.dataframe(res.preview_head, use_container_width=True)


    st.subheader("Final Output")
    st.dataframe(out.df.head(20), use_container_width=True)
    st.write(f"Final shape: {out.df.shape}")

    st.download_button(
        "Download final_features.csv",
        data=out.df.to_csv(index=False),
        file_name="final_features.csv",
        mime="text/csv",
    )
