import streamlit as st
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

st.set_page_config(page_title="Jane Street EDA", layout="wide")
st.title("Jane Street Real-Time Market Data — EDA")

ROOT_DIR = st.sidebar.text_input(
    "Dataset root path",
    value="/kaggle/input/jane-street-real-time-market-data-forecasting",
)

PARTITION_OPTIONS = list(range(10))


def load_parquet(root, partition_id):
    path = f"{root}/train.parquet/partition_id={partition_id}/part-0.parquet"
    return pl.read_parquet(path)


section = st.sidebar.radio(
    "Section",
    [
        "Features Metadata",
        "Responders Metadata",
        "Sample Submission",
        "Missing Values",
        "Feature Correlations",
        "Responder Distributions",
        "Responder Correlations",
        "Symbol ID Distribution",
        "Date ID Distribution",
    ],
)


if section == "Features Metadata":
    st.header("Features Metadata")
    fpath = f"{ROOT_DIR}/features.csv"
    if not os.path.exists(fpath):
        st.error(f"File not found: {fpath}")
        st.stop()

    features = pd.read_csv(fpath)
    st.dataframe(features, use_container_width=True)

    st.subheader("Feature Tag Bitmap")
    fig, ax = plt.subplots(figsize=(20, 6))
    ax.imshow(features.iloc[:, 1:].T.values, cmap="gray", aspect="auto")
    ax.set_xlabel("feature_00 ~ feature_78")
    ax.set_ylabel("tag_0 ~ tag_16")
    ax.set_yticks(np.arange(17))
    ax.set_xticks(np.arange(79))
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Tag Correlation Heatmap")
    tag_cols = [f"tag_{no}" for no in range(17)]
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    sns.heatmap(features[tag_cols].T.corr(), square=True, cmap="jet", ax=ax2)
    st.pyplot(fig2)
    plt.close(fig2)


elif section == "Responders Metadata":
    st.header("Responders Metadata")
    rpath = f"{ROOT_DIR}/responders.csv"
    if not os.path.exists(rpath):
        st.error(f"File not found: {rpath}")
        st.stop()

    responders = pd.read_csv(rpath)
    st.dataframe(responders, use_container_width=True)

    st.subheader("Responder Tag Correlation Heatmap")
    tag_cols = [f"tag_{no}" for no in range(5)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        responders[tag_cols].T.corr(),
        annot=True,
        square=True,
        cmap="jet",
        ax=ax,
    )
    ax.set_xlabel("responder_0 ~ responder_8")
    ax.set_ylabel("responder_0 ~ responder_8")
    st.pyplot(fig)
    plt.close(fig)


elif section == "Sample Submission":
    st.header("Sample Submission")
    spath = f"{ROOT_DIR}/sample_submission.csv"
    if not os.path.exists(spath):
        st.error(f"File not found: {spath}")
        st.stop()

    sub = pd.read_csv(spath)
    st.write(f"Shape: {sub.shape}")
    st.dataframe(sub, use_container_width=True)


elif section == "Missing Values":
    st.header("Missing Values per Partition")
    partition_id = st.sidebar.selectbox("Partition ID", PARTITION_OPTIONS)

    path = f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    with st.spinner("Loading parquet..."):
        train = pl.read_parquet(path)

    supervised = train.filter(pl.col("responder_6").is_not_null())
    missing_count = (
        supervised.null_count()
        .transpose(include_header=True, header_name="feature", column_names=["null_count"])
        .sort("null_count", descending=True)
        .with_columns((pl.col("null_count") / len(supervised)).alias("null_ratio"))
    )

    mc = missing_count.to_pandas()
    st.write(f"Samples with target: {len(supervised):,}")
    st.dataframe(mc, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 22))
    y = np.arange(len(mc))
    ax.barh(y, mc["null_ratio"], color="coral", label="missing")
    ax.barh(y, 1 - mc["null_ratio"], left=mc["null_ratio"], color="darkseagreen", label="available")
    ax.set_yticks(y)
    ax.set_yticklabels(mc["feature"], fontsize=7)
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.set_xlim(0, 1)
    ax.legend()
    ax.set_title(f"Missing values — partition {partition_id}")
    st.pyplot(fig)
    plt.close(fig)


elif section == "Feature Correlations":
    st.header("Feature Correlation Heatmap (79 features)")
    partition_id = st.sidebar.selectbox("Partition ID", PARTITION_OPTIONS)

    path = f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    with st.spinner("Loading parquet..."):
        train = pl.read_parquet(path)

    feat_cols = [f"feature_{i:02d}" for i in range(79)]
    corr = train[feat_cols].to_pandas().corr()

    fig, ax = plt.subplots(figsize=(18, 16))
    sns.heatmap(corr, square=True, cmap="jet", ax=ax, xticklabels=4, yticklabels=4)
    ax.set_xlabel("feature_00 ~ feature_78")
    ax.set_ylabel("feature_00 ~ feature_78")
    st.pyplot(fig)
    plt.close(fig)


elif section == "Responder Distributions":
    st.header("Responder Distributions")
    partition_id = st.sidebar.selectbox("Partition ID", PARTITION_OPTIONS)

    path = f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    with st.spinner("Loading parquet..."):
        train = pl.read_parquet(path)

    selected = st.multiselect(
        "Select responders", [f"responder_{i}" for i in range(9)], default=["responder_6"]
    )

    for col in selected:
        series = train[col].drop_nulls()
        mean_ = series.mean()
        sgm_ = np.sqrt(series.var())
        min_ = series.min()
        max_ = series.max()

        st.subheader(col)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Mean", f"{mean_:.4f}")
        c2.metric("Std", f"{sgm_:.4f}")
        c3.metric("Min", f"{min_:.4f}")
        c4.metric("Max", f"{max_:.4f}")

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(series.to_numpy(), bins=40, color="steelblue", edgecolor="white")
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig)


elif section == "Responder Correlations":
    st.header("Responder Correlation Heatmap")
    partition_id = st.sidebar.selectbox("Partition ID", PARTITION_OPTIONS)

    path = f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    with st.spinner("Loading parquet..."):
        train = pl.read_parquet(path)

    resp_cols = [f"responder_{i}" for i in range(9)]
    corr = train[resp_cols].to_pandas().corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, annot=True, square=True, cmap="jet", ax=ax, fmt=".2f")
    ax.set_xlabel("responder_0 ~ responder_8")
    ax.set_ylabel("responder_0 ~ responder_8")
    st.pyplot(fig)
    plt.close(fig)


elif section == "Symbol ID Distribution":
    st.header("Symbol ID Distribution")
    partition_id = st.sidebar.selectbox("Partition ID", PARTITION_OPTIONS)

    path = f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    with st.spinner("Loading parquet..."):
        train = pl.read_parquet(path)

    sid = train["symbol_id"]
    st.write(f"symbol_id range: {sid.min()} – {sid.max()}")

    bins = int(sid.max() - sid.min() + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(sid.to_numpy(), bins=min(bins, 200), color="mediumpurple", edgecolor="white")
    ax.set_xlabel("symbol_id")
    ax.set_ylabel("frequency")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)


elif section == "Date ID Distribution":
    st.header("Date ID Distribution")
    partition_id = st.sidebar.selectbox("Partition ID", PARTITION_OPTIONS)

    path = f"{ROOT_DIR}/train.parquet/partition_id={partition_id}/part-0.parquet"
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()

    with st.spinner("Loading parquet..."):
        train = pl.read_parquet(path)

    did = train["date_id"]
    st.write(f"date_id range: {did.min()} – {did.max()}")

    bins = int(did.max() - did.min() + 1)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(did.to_numpy(), bins=min(bins, 300), color="teal", edgecolor="white")
    ax.set_xlabel("date_id")
    ax.set_ylabel("frequency")
    ax.grid(True)
    st.pyplot(fig)
    plt.close(fig)
