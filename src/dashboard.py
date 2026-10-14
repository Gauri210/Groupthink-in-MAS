"""
Groupthink MAS – Streamlit Dashboard
Run: streamlit run dashboard.py  (from src/)
Reads results/processed/*.csv produced by analyze_results.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

st.set_page_config(page_title="Groupthink MAS", layout="wide")
st.title("Groupthink in Multi-Agent LLM Systems")

# ── paths ──────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(BASE, "..", "results", "processed")

def load(name):
    return pd.read_csv(os.path.join(PROC, name))

try:
    cond = load("summary_by_condition.csv").set_index("condition")
    cat  = load("summary_by_category.csv")
    perq = load("per_question_comparison.csv")
    combined = load("combined_results.csv")
except FileNotFoundError as e:
    st.error(f"CSV not found – run analyze_results.py first.\n{e}")
    st.stop()

CONDITIONS = ["solo_baseline", "group_deliberation", "group_with_contrarian"]
cond = cond.reindex(CONDITIONS)

# ── helpers ────────────────────────────────────────────────────────────
def bar(ax, vals, labels, title, ylabel, colors):
    ax.bar(labels, vals, color=colors)
    ax.set_title(title); ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1); ax.set_xticklabels(labels, rotation=15, ha="right")

COLORS = ["#4C72B0", "#DD8452", "#55A868"]
SHORT  = ["Solo", "Group", "Contrarian"]

# ── row 1: 3 metric charts ─────────────────────────────────────────────
st.subheader("Condition-level metrics")
c1, c2, c3 = st.columns(3)

for col, metric, title in [
    (c1, "is_correct",  "Accuracy"),
    (c2, "flipped",     "Flip Rate"),
    (c3, "rescued",     "Rescue Rate"),
]:
    fig, ax = plt.subplots(figsize=(3.5, 3))
    bar(ax, cond[metric].values, SHORT, title, "Rate", COLORS)
    col.pyplot(fig, use_container_width=True)
    plt.close(fig)

# ── row 2: category accuracy ───────────────────────────────────────────
st.subheader("Accuracy by category")
fig, ax = plt.subplots(figsize=(7, 3.5))
cats = cat["category"].tolist()
x = range(len(cats))
w = 0.25
for i, (col, label) in enumerate(zip(CONDITIONS, SHORT)):
    ax.bar([v + i*w for v in x], cat[col].values, w, label=label, color=COLORS[i])
ax.set_xticks([v + w for v in x]); ax.set_xticklabels(cats)
ax.set_ylim(0, 1); ax.set_ylabel("Accuracy"); ax.legend()
st.pyplot(fig, use_container_width=True)
plt.close(fig)

# ── row 3: per-question heatmap ────────────────────────────────────────
# ── row 3: interesting questions ──────────────────────────────────────
st.subheader("Notable questions")

solo_col = "solo_baseline"
grp_col  = "group_deliberation"
con_col  = "group_with_contrarian"

solo_wrong_grp_right = perq[(perq[solo_col]==0) & (perq[grp_col]==1)]["question_id"].tolist()
solo_right_grp_wrong = perq[(perq[solo_col]==1) & (perq[grp_col]==0)]["question_id"].tolist()
con_rescued          = perq[(perq[grp_col]==0)  & (perq[con_col]==1)]["question_id"].tolist()
con_hurt             = perq[(perq[grp_col]==1)  & (perq[con_col]==0)]["question_id"].tolist()

c1, c2 = st.columns(2)
c1.metric("Group rescued solo errors", f"Q{solo_wrong_grp_right}" if solo_wrong_grp_right else "None")
c2.metric("Group introduced errors",   f"Q{solo_right_grp_wrong}" if solo_right_grp_wrong else "None")
c1.metric("Contrarian rescued group",  f"Q{con_rescued}" if con_rescued else "None")
c2.metric("Contrarian hurt group",     f"Q{con_hurt}"    if con_hurt    else "None")

# ── row 4: summary table ───────────────────────────────────────────────
st.subheader("Summary table")
rate_cols = ["is_correct", "flipped", "rescued", "no_consensus"]
st.dataframe(
    cond.style
    .format({c: "{:.1%}" for c in rate_cols if c in cond.columns})
    .highlight_max(subset=rate_cols, axis=0, color="#1a662b")
    .highlight_min(subset=rate_cols, axis=0, color="#69151c")
)

# ── row 5: raw data explorer ───────────────────────────────────────────
with st.expander("Raw data explorer"):
    filt = st.selectbox("Condition", ["all"] + CONDITIONS)
    df = combined if filt == "all" else combined[combined.condition == filt]
    st.dataframe(df.drop(columns=["history"], errors="ignore"), height=300)