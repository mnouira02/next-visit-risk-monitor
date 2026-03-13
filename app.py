import os
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from preprocessing_universal import UniversalSDTMPreprocessor, RiskPredictionDataset
from model import NextVisitRiskTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RISK_NAMES = {0: "Low", 1: "Medium", 2: "High"}
RISK_COLORS = {"Low": "#00CC96", "Medium": "#FFA15A", "High": "#EF553B"}

st.set_page_config(
    page_title="Visit Risk Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1rem; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    .stMetric {
        background-color: rgba(255,255,255,0.02);
        padding: 0.4rem 0.6rem;
        border-radius: 0.5rem;
        min-height: 100px;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem !important;
        color: #BFC7D5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def format_visit(v):
    v = float(v)
    if v.is_integer():
        return f"Visit {int(v)}"
    return f"Visit {v:.1f}"


def format_visit_short(v):
    v = float(v)
    if v.is_integer():
        return f"V{int(v)}"
    return f"V{v:.1f}"


def shorten_label(label, max_len=34):
    label = str(label).strip()
    if len(label) <= max_len:
        return label
    return textwrap.shorten(label, width=max_len, placeholder="...")


def risk_html(risk):
    colors = {
        "Low": "#00CC96",
        "Medium": "#FFA15A",
        "High": "#EF553B"
    }
    color = colors.get(risk, "#D0D0D0")
    return f"<span style='color:{color}; font-weight:700;'>{risk}</span>"


def build_visit_option_label(target_visit, prediction_error):
    visit_text = format_visit(target_visit)
    if pd.notna(prediction_error):
        return f"{visit_text} | error {prediction_error:.0f}d"
    return f"{visit_text} | no actual"


def intervention_text(row):
    pred_risk = row["Predicted Risk"]

    if pred_risk == "High":
        return (
            "Immediate outreach recommended: confirm patient availability, identify transport or scheduling barriers, "
            "coordinate with the site on contingency scheduling, and escalate if the visit affects key endpoint capture."
        )

    if pred_risk == "Medium":
        return (
            "Proactive follow-up recommended: send reminders, confirm the visit booking with the site, and watch for new "
            "signals such as missed assessments or unscheduled activity."
        )

    return (
        "No immediate escalation suggested: continue routine follow-up and monitor for any new timing signals before the visit."
    )


def generate_visit_insight(row, show_actuals):
    visit_label = format_visit(row["Target Visit"])
    pred_risk = row["Predicted Risk"]
    p_high = row["P_High"]
    p_med = row["P_Medium"]
    planned_day = row["Target Planned Day"]
    predicted_day = row["Predicted Day"]
    gap_days = row["Days Until Target"]

    base = (
        f"{visit_label}: the model predicts {risk_html(pred_risk)} risk of missing the visit window, "
        f"with probabilities low/medium/high = {row['P_Low']:.0%}/{p_med:.0%}/{p_high:.0%}. "
        f"The planned day is {planned_day:.0f}, and the model-implied predicted day is {predicted_day:.0f}. "
        f"The current historical runway to target is {gap_days:.0f} days."
    )

    if show_actuals and pd.notna(row["Target Actual Day"]):
        actual_day = row["Target Actual Day"]
        diff = actual_day - predicted_day
        actual_risk_html = risk_html(row["Actual Risk"])
        window_flag = "within window" if row["Within Window"] else "out of window"
        base += (
            f" The observed visit occurred on day {actual_day:.0f}, which is {diff:+.0f} days relative to the "
            f"predicted day. The actual outcome was {actual_risk_html} risk and was ultimately {window_flag}."
        )

    return base


def build_history_dataframe(processor, x_cat, x_num, x_time, x_visit, x_planned, example_idx):
    vocab_rev = {v: k for k, v in processor.vocab.items()}
    rev_visit_map = {v: k for k, v in processor.visit_map.items()}

    cat = x_cat[example_idx].cpu().numpy()
    num = x_num[example_idx].cpu().numpy()
    time = x_time[example_idx].cpu().numpy()
    visit = x_visit[example_idx].cpu().numpy()
    planned = x_planned[example_idx].cpu().numpy()

    mask = cat != 0
    cat = cat[mask]
    num = num[mask]
    time = time[mask]
    visit = visit[mask]
    planned = planned[mask]

    tokens = [vocab_rev.get(int(c), "UNK") for c in cat]
    labels = [processor.label_map.get(t, t.split("_")[-1]) for t in tokens]
    domains = [t.split("_")[0] if "_" in t else "UNK" for t in tokens]
    visit_nums = [rev_visit_map.get(int(v), v) for v in visit]

    history_df = pd.DataFrame({
        "Event Day": time,
        "Value": num,
        "Visit Number": visit_nums,
        "Planned Day": planned,
        "Token": tokens,
        "Label": labels,
        "Domain": domains
    }).sort_values(["Event Day", "Domain", "Label"]).reset_index(drop=True)

    return history_df


def plot_population_cluster(site_df, selected_df):
    site_custom = np.array(
        list(
            zip(
                site_df["USUBJID"].astype(str),
                [format_visit(v) for v in site_df["Target Visit"]],
                site_df["Predicted Risk"].astype(str),
                [f"{p:.1%}" for p in site_df["P_High"]]
            )
        ),
        dtype=object
    )

    sel_custom = np.array(
        list(
            zip(
                selected_df["USUBJID"].astype(str),
                [format_visit(v) for v in selected_df["Target Visit"]],
                selected_df["Predicted Risk"].astype(str),
                [f"{p:.1%}" for p in selected_df["P_High"]]
            )
        ),
        dtype=object
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=site_df["x"],
            y=site_df["y"],
            mode="markers",
            marker=dict(
                size=7,
                color="rgba(120,120,120,0.22)"
            ),
            customdata=site_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Predicted Risk: %{customdata[2]}<br>"
                "High Risk Prob: %{customdata[3]}<extra></extra>"
            ),
            name="Comparison Population",
            showlegend=False
        )
    )

    highlight_colors = [RISK_COLORS[r] for r in selected_df["Predicted Risk"]]

    fig.add_trace(
        go.Scatter(
            x=selected_df["x"],
            y=selected_df["y"],
            mode="markers+text",
            marker=dict(
                size=14,
                color=highlight_colors,
                line=dict(width=1.5, color="black"),
                opacity=0.95
            ),
            text=[format_visit_short(v) for v in selected_df["Target Visit"]],
            textposition="top center",
            customdata=sel_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "Predicted Risk: %{customdata[2]}<br>"
                "High Risk Prob: %{customdata[3]}<extra></extra>"
            ),
            name="Selected",
            showlegend=False
        )
    )

    fig.update_layout(
        height=430,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Population Context",
        xaxis_title="Embedding Dimension 1",
        yaxis_title="Embedding Dimension 2",
        plot_bgcolor="white",
        paper_bgcolor="white"
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.20)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(180,180,180,0.20)", zeroline=False)

    return fig


def plot_visit_prediction_chart(patient_visits_df, show_actuals):
    df = patient_visits_df.copy().reset_index(drop=True)
    df["y"] = list(range(len(df), 0, -1))

    fig = go.Figure()

    for _, row in df.iterrows():
        y = row["y"]
        planned = float(row["Target Planned Day"])
        window = float(row["Window Days"])
        predicted = float(row["Predicted Day"])

        fig.add_shape(
            type="line",
            x0=planned - window,
            y0=y,
            x1=planned + window,
            y1=y,
            line=dict(color="rgba(0, 204, 150, 0.35)", width=14)
        )

        if show_actuals and pd.notna(row["Target Actual Day"]):
            actual = float(row["Target Actual Day"])
            arrow_color = "#EF553B" if actual > predicted else "#4F6BED"

            fig.add_annotation(
                x=actual,
                y=y,
                ax=predicted,
                ay=y,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.2,
                arrowwidth=2,
                arrowcolor=arrow_color,
                opacity=0.9
            )

    planned_custom = np.array(
        list(
            zip(
                [format_visit(v) for v in df["Target Visit"]],
                [f"±{w:.0f} days" for w in df["Window Days"]]
            )
        ),
        dtype=object
    )

    pred_custom = np.array(
        list(
            zip(
                [format_visit(v) for v in df["Target Visit"]],
                df["Predicted Risk"].astype(str),
                [f"{p:.1%}" for p in df["P_High"]]
            )
        ),
        dtype=object
    )

    fig.add_trace(
        go.Scatter(
            x=df["Target Planned Day"],
            y=df["y"],
            mode="markers",
            marker=dict(color="#FFA15A", size=11, symbol="diamond-open"),
            name="Planned Day",
            customdata=planned_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Planned Day: %{x:.0f}<br>"
                "Window: %{customdata[1]}<extra></extra>"
            )
        )
    )

    fig.add_trace(
        go.Scatter(
            x=df["Predicted Day"],
            y=df["y"],
            mode="markers",
            marker=dict(
                color=[RISK_COLORS[r] for r in df["Predicted Risk"]],
                size=13,
                symbol="x",
                line=dict(width=1.5, color="black")
            ),
            name="Predicted Day",
            customdata=pred_custom,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Predicted Day: %{x:.0f}<br>"
                "Predicted Risk: %{customdata[1]}<br>"
                "High Risk Prob: %{customdata[2]}<extra></extra>"
            )
        )
    )

    if show_actuals:
        actual_df = df[df["Target Actual Day"].notna()].copy()
        if not actual_df.empty:
            actual_colors = ["#00CC96" if bool(v) else "#EF553B" for v in actual_df["Within Window"]]
            actual_custom = np.array(
                list(
                    zip(
                        [format_visit(v) for v in actual_df["Target Visit"]],
                        actual_df["Actual Risk"].astype(str),
                        [f"{d:+.0f} days" for d in actual_df["Diff Days"].fillna(0)]
                    )
                ),
                dtype=object
            )

            fig.add_trace(
                go.Scatter(
                    x=actual_df["Target Actual Day"],
                    y=actual_df["y"],
                    mode="markers",
                    marker=dict(
                        color=actual_colors,
                        size=12,
                        symbol="circle",
                        line=dict(width=1.2, color="black")
                    ),
                    name="Actual Day",
                    customdata=actual_custom,
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "Actual Day: %{x:.0f}<br>"
                        "Actual Risk: %{customdata[1]}<br>"
                        "Diff vs Plan: %{customdata[2]}<extra></extra>"
                    )
                )
            )

    fig.update_layout(
        height=max(320, 120 + 82 * len(df)),
        margin=dict(l=10, r=10, t=80, b=10),
        title=dict(
            text="Visit Prediction vs Actual",
            x=0.5,
            xanchor="center"
        ),
        xaxis_title="Study Day",
        yaxis_title="Visit",
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.03,
            xanchor="center",
            x=0.5
        )
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(180,180,180,0.20)", zeroline=False)
    fig.update_yaxes(
        tickmode="array",
        tickvals=df["y"].tolist(),
        ticktext=[format_visit(v) for v in df["Target Visit"]],
        showgrid=False,
        zeroline=False
    )

    return fig


@st.cache_resource
def load_system():
    default_window_days = 7

    processor = UniversalSDTMPreprocessor()

    if not os.path.exists("data"):
        st.error("Data folder not found.")
        st.stop()

    try:
        events_flat = processor.load_data("data")
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        st.stop()

    processor.fit(events_flat)

    tensors, df_examples = processor.build_risk_dataset(
        events_df=events_flat,
        default_window_days=default_window_days,
        medium_multiplier=2.0,
        min_history_events=5
    )

    (
        X_cat,
        X_num,
        X_time,
        X_visit,
        X_planned,
        X_target_visit,
        X_target_planned,
        y
    ) = tensors

    dataset = RiskPredictionDataset(
        X_cat,
        X_num,
        X_time,
        X_visit,
        X_planned,
        X_target_visit,
        X_target_planned,
        y
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    vocab_size = len(processor.vocab)
    max_visits = max(processor.visit_map.values()) + 16

    model = NextVisitRiskTransformer(
        vocab_size=vocab_size,
        max_visits=max_visits
    ).to(DEVICE)

    class_counts = torch.bincount(y, minlength=3).float()
    class_weights = class_counts.sum() / class_counts.clamp(min=1)
    class_weights = class_weights / class_weights.mean()

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    progress = st.progress(0)

    for epoch in range(18):
        for batch in loader:
            (
                b_cat,
                b_num,
                b_time,
                b_visit,
                b_planned,
                b_target_visit,
                b_target_planned,
                b_y
            ) = [x.to(DEVICE) for x in batch]

            logits, _ = model(
                b_cat,
                b_num,
                b_time,
                b_visit,
                b_planned,
                b_target_visit,
                b_target_planned
            )

            loss = criterion(logits, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        progress.progress((epoch + 1) / 18)

    progress.empty()

    model.eval()
    with torch.no_grad():
        logits, embeddings = model(
            X_cat.to(DEVICE),
            X_num.to(DEVICE),
            X_time.to(DEVICE),
            X_visit.to(DEVICE),
            X_planned.to(DEVICE),
            X_target_visit.to(DEVICE),
            X_target_planned.to(DEVICE)
        )

        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred_class = probs.argmax(axis=1)
        emb_np = embeddings.cpu().numpy()

    df_examples = df_examples.copy()
    df_examples["Actual Class Id"] = y.cpu().numpy()
    df_examples["Pred Class Id"] = pred_class
    df_examples["Predicted Risk"] = [RISK_NAMES[int(v)] for v in pred_class]
    df_examples["P_Low"] = probs[:, 0]
    df_examples["P_Medium"] = probs[:, 1]
    df_examples["P_High"] = probs[:, 2]
    df_examples["Correct"] = df_examples["Actual Class Id"] == df_examples["Pred Class Id"]

    diff_source = df_examples[df_examples["Diff Days"].notna()].copy()
    mean_diff_by_class = (
        diff_source.groupby("Actual Class Id")["Diff Days"]
        .mean()
        .reindex([0, 1, 2])
        .fillna(0.0)
        .values
    )

    predicted_diff = probs @ mean_diff_by_class
    df_examples["Predicted Diff Days"] = predicted_diff
    df_examples["Predicted Day"] = df_examples["Target Planned Day"] + df_examples["Predicted Diff Days"]
    df_examples["Prediction Error"] = np.where(
        df_examples["Target Actual Day"].notna(),
        np.abs(df_examples["Predicted Day"] - df_examples["Target Actual Day"]),
        np.nan
    )

    pca = PCA(n_components=2)
    pca_res = pca.fit_transform(emb_np)
    df_examples["x"] = pca_res[:, 0]
    df_examples["y"] = pca_res[:, 1]

    tensors_out = (
        X_cat,
        X_num,
        X_time,
        X_visit,
        X_planned,
        X_target_visit,
        X_target_planned,
        y
    )

    return model, processor, events_flat, tensors_out, df_examples, default_window_days


if "risk_data" not in st.session_state:
    with st.spinner("🚀 Booting Clinical Timewarp Risk Engine..."):
        (
            model,
            processor,
            df_events,
            tensors,
            df_examples,
            default_window_days
        ) = load_system()
        st.session_state["risk_data"] = (
            model,
            processor,
            df_events,
            tensors,
            df_examples,
            default_window_days
        )
else:
    (
        model,
        processor,
        df_events,
        tensors,
        df_examples,
        default_window_days
    ) = st.session_state["risk_data"]

(
    X_cat,
    X_num,
    X_time,
    X_visit,
    X_planned,
    X_target_visit,
    X_target_planned,
    y
) = tensors


st.sidebar.title("🛡️ Next Visit Risk Prediction")

sites = sorted(df_examples["Site"].dropna().astype(str).unique().tolist())
site_options = ["-- Select Site --"] + sites
selected_site = st.sidebar.selectbox("Site", site_options, index=0)

show_actuals = st.sidebar.checkbox("Show actual visit day", value=True)

if selected_site == "-- Select Site --":
    st.stop()

site_df = df_examples[df_examples["Site"].astype(str) == selected_site].copy()

patient_sort_mode = st.sidebar.selectbox(
    "Patient Order",
    ["Closest prediction first", "Worst prediction first", "Visit order"],
    index=0
)

visit_sort_mode = st.sidebar.selectbox(
    "Visit Order",
    ["Closest prediction first", "Worst prediction first", "Visit order"],
    index=0
)

patient_rank_df = site_df.copy()
patient_rank_df["Prediction Error Filled"] = patient_rank_df["Prediction Error"].fillna(999999)

patient_summary = (
    patient_rank_df.groupby("USUBJID", as_index=False)
    .agg(
        Best_Error=("Prediction Error Filled", "min"),
        Mean_Error=("Prediction Error Filled", "mean")
    )
)

if patient_sort_mode == "Closest prediction first":
    ordered_patients = patient_summary.sort_values(
        ["Best_Error", "Mean_Error", "USUBJID"],
        ascending=[True, True, True]
    )["USUBJID"].tolist()
elif patient_sort_mode == "Worst prediction first":
    ordered_patients = patient_summary.sort_values(
        ["Best_Error", "Mean_Error", "USUBJID"],
        ascending=[False, False, True]
    )["USUBJID"].tolist()
else:
    ordered_patients = sorted(site_df["USUBJID"].dropna().astype(str).unique().tolist())

patient_options = ["-- Select Patient --"] + ordered_patients
selected_patient = st.sidebar.selectbox("Patient", patient_options, index=0)

if selected_patient == "-- Select Patient --":
    st.stop()

patient_all_visits_df = (
    site_df[site_df["USUBJID"].astype(str) == selected_patient]
    .copy()
)

patient_all_visits_df["Prediction Error Filled"] = patient_all_visits_df["Prediction Error"].fillna(999999)

if visit_sort_mode == "Closest prediction first":
    patient_all_visits_df = patient_all_visits_df.sort_values(
        ["Prediction Error Filled", "Target Visit"],
        ascending=[True, True]
    ).reset_index(drop=True)
elif visit_sort_mode == "Worst prediction first":
    patient_all_visits_df = patient_all_visits_df.sort_values(
        ["Prediction Error Filled", "Target Visit"],
        ascending=[False, True]
    ).reset_index(drop=True)
else:
    patient_all_visits_df = patient_all_visits_df.sort_values("Target Visit").reset_index(drop=True)

visit_options = ["All Visits"] + [
    build_visit_option_label(v, e)
    for v, e in zip(patient_all_visits_df["Target Visit"], patient_all_visits_df["Prediction Error"])
]
selected_visit_label = st.sidebar.selectbox("Visit", visit_options, index=0)

if selected_visit_label == "All Visits":
    selected_visits_df = patient_all_visits_df.copy()
else:
    selected_visits_df = patient_all_visits_df[
        [
            build_visit_option_label(v, e) == selected_visit_label
            for v, e in zip(patient_all_visits_df["Target Visit"], patient_all_visits_df["Prediction Error"])
        ]
    ].copy()

selected_example_idx = int(selected_visits_df.iloc[0]["Example Index"])

if st.sidebar.button("🔄 Force Reload"):
    if os.path.exists("sdtm_data_cache.pkl"):
        os.remove("sdtm_data_cache.pkl")
    st.cache_resource.clear()
    if "risk_data" in st.session_state:
        del st.session_state["risk_data"]
    st.rerun()

label_len = st.sidebar.slider("Raw Event Label Length", 18, 60, 34, 2)


c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2])
with c1:
    st.title(f"Patient: {selected_patient}")
    if selected_visit_label == "All Visits":
        st.caption(f"Site: {selected_site} | Showing all predicted visits")
    else:
        st.caption(f"Site: {selected_site} | Showing {selected_visit_label}")
with c2:
    st.metric("Visits displayed", f"{len(selected_visits_df)}")
with c3:
    st.metric("Average high-risk probability", f"{selected_visits_df['P_High'].mean():.1%}")
with c4:
    best_error = selected_visits_df["Prediction Error"].min()
    if pd.notna(best_error):
        st.metric("Closest prediction error", f"{best_error:.0f}d")
    else:
        st.metric("Closest prediction error", "N/A")

st.caption(
    "Predicted day is a model-implied estimate derived from the visit-risk probabilities, used here as a visual anchor for the predicted-to-actual arrow."
)

cluster_fig = plot_population_cluster(site_df=site_df, selected_df=selected_visits_df)
st.plotly_chart(cluster_fig, use_container_width=True)

visit_fig = plot_visit_prediction_chart(
    patient_visits_df=selected_visits_df,
    show_actuals=show_actuals
)
st.plotly_chart(visit_fig, use_container_width=True)

st.markdown("## Insights")
for _, row in selected_visits_df.iterrows():
    st.markdown(f"**{format_visit(row['Target Visit'])}**")
    st.markdown(generate_visit_insight(row, show_actuals), unsafe_allow_html=True)
    st.caption(intervention_text(row))
    st.divider()

with st.expander("Raw historical events used before the selected visit snapshot(s)"):
    history_df = build_history_dataframe(
        processor=processor,
        x_cat=X_cat,
        x_num=X_num,
        x_time=X_time,
        x_visit=X_visit,
        x_planned=X_planned,
        example_idx=selected_example_idx
    )

    if not history_df.empty:
        history_df["Label"] = history_df["Label"].apply(lambda x: shorten_label(x, label_len))

    st.dataframe(
        history_df[["Event Day", "Visit Number", "Domain", "Label", "Token", "Value", "Planned Day"]],
        use_container_width=True
    )

with st.expander(f"Raw SDTM data for {selected_patient}"):
    subj_data = df_events[df_events["USUBJID"].astype(str) == selected_patient].copy()
    st.dataframe(
        subj_data[["DTC", "VISITNUM", "TOKEN", "HUMAN_LABEL", "VAL", "DY", "PLANNED_DY"]],
        use_container_width=True
    )
