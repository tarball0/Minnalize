"""
NexusGuard - Real-Time Anomaly Detection Dashboard
Interactive Streamlit dashboard for monitoring, analysis, and alerting.
"""

import sys
import os

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure project root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from nexusguard.config import NexusGuardConfig
from nexusguard.data import ActivityLogGenerator
from nexusguard.engine.pipeline import NexusGuardPipeline


# ──────────────────────────── Page Config ─────────────────────────────
st.set_page_config(
    page_title="NexusGuard - Anomaly Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────── Custom CSS ──────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
    }
    .risk-severe { color: #ff1744; font-weight: bold; }
    .risk-critical { color: #ff5722; font-weight: bold; }
    .risk-high { color: #ff9800; font-weight: bold; }
    .risk-medium { color: #ffc107; }
    .risk-low { color: #4caf50; }
    .stMetric { background-color: #0e1117; border-radius: 10px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Session State ───────────────────────────
def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "events_df" not in st.session_state:
        st.session_state.events_df = None
    if "scores_df" not in st.session_state:
        st.session_state.scores_df = None
    if "alerts" not in st.session_state:
        st.session_state.alerts = []
    if "metrics" not in st.session_state:
        st.session_state.metrics = {}
    if "eval_metrics" not in st.session_state:
        st.session_state.eval_metrics = {}
    if "raw_data" not in st.session_state:
        st.session_state.raw_data = None

init_session()


# ──────────────────────────── Sidebar ─────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    num_events = st.slider("Normal Events", 1000, 20000, 5000, step=1000)
    num_anomalies = st.slider("Injected Anomalies", 50, 2000, 300, step=50)
    num_users = st.slider("Number of Users", 10, 200, 50, step=10)

    st.markdown("---")
    st.markdown("### Model Weights")
    w_ae = st.slider("Autoencoder", 0.0, 1.0, 0.35, step=0.05)
    w_lstm = st.slider("LSTM Sequence", 0.0, 1.0, 0.30, step=0.05)
    w_if = st.slider("Isolation Forest", 0.0, 1.0, 0.20, step=0.05)
    w_beh = st.slider("Behavioral", 0.0, 1.0, 0.15, step=0.05)

    st.markdown("---")
    threshold = st.slider("Anomaly Threshold", 0.3, 0.95, 0.65, step=0.05)

    st.markdown("---")
    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        config = NexusGuardConfig()
        config.data.num_normal_events = num_events
        config.data.num_anomalous_events = num_anomalies
        config.data.num_users = num_users
        config.model.ensemble_weights = {
            "autoencoder": w_ae,
            "lstm": w_lstm,
            "isolation_forest": w_if,
            "behavioral": w_beh,
        }
        config.model.anomaly_threshold = threshold

        with st.spinner("Generating synthetic activity logs..."):
            generator = ActivityLogGenerator(config)
            data = generator.generate()
            st.session_state.raw_data = data

        with st.spinner("Training NexusGuard models..."):
            pipeline = NexusGuardPipeline(config)
            train_metrics = pipeline.train(data)
            st.session_state.pipeline = pipeline
            st.session_state.metrics = train_metrics

        with st.spinner("Running anomaly detection..."):
            events_df, scores_df, alerts = pipeline.detect_and_alert(data)
            st.session_state.events_df = events_df
            st.session_state.scores_df = scores_df
            st.session_state.alerts = alerts

        with st.spinner("Evaluating performance..."):
            eval_metrics = pipeline.evaluate(data)
            st.session_state.eval_metrics = eval_metrics

        st.success("Pipeline complete!")


# ──────────────────────────── Main Content ────────────────────────────
st.markdown('<p class="main-header">NexusGuard</p>', unsafe_allow_html=True)
st.markdown("**Adaptive Behavioral Anomaly Detection Engine** — Deep Learning · LSTM · Isolation Forest · Graph Analysis")
st.markdown("---")

if st.session_state.events_df is None:
    st.info("👈 Configure parameters and click **Run Full Pipeline** to begin analysis.")
    st.markdown("""
    ### How it works
    NexusGuard uses an ensemble of advanced ML models to detect anomalous behavior in system activity logs:

    | Model | Technique | What it Catches |
    |-------|-----------|----------------|
    | **Deep Autoencoder** | Reconstruction error | Events that don't fit learned normal patterns |
    | **LSTM + Attention** | Sequence prediction | Unusual temporal sequences of events |
    | **Isolation Forest** | Random partitioning | Statistical outliers in feature space |
    | **Behavioral Profiler** | Per-user statistical models | Deviation from individual user baselines |
    | **Graph Analyzer** | Network community detection | Unusual communication patterns |
    """)
    st.stop()


events_df = st.session_state.events_df
scores_df = st.session_state.scores_df
alerts = st.session_state.alerts
metrics = st.session_state.metrics
eval_metrics = st.session_state.eval_metrics

# ──────────────────── Tab Layout ──────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 Anomaly Explorer", "⚠️ Alerts", "📈 Model Performance", "🔬 Deep Dive"
])

# ═══════════════════════════ TAB 1: OVERVIEW ══════════════════════════
with tab1:
    col1, col2, col3, col4, col5 = st.columns(5)
    total_events = len(events_df)
    total_anomalies = int(scores_df["is_anomaly"].sum())
    total_alerts = len(alerts)
    avg_score = float(scores_df["ensemble_score"].mean())

    col1.metric("Total Events", f"{total_events:,}")
    col2.metric("Anomalies Detected", f"{total_anomalies:,}", delta=f"{total_anomalies/total_events*100:.1f}%")
    col3.metric("Alerts Generated", f"{total_alerts:,}")
    col4.metric("Avg Anomaly Score", f"{avg_score:.3f}")
    col5.metric("F1 Score", f"{eval_metrics.get('f1_score', 0):.3f}")

    st.markdown("---")
    col_left, col_right = st.columns(2)

    with col_left:
        # Score distribution
        fig = px.histogram(
            scores_df, x="ensemble_score", nbins=50,
            color_discrete_sequence=["#667eea"],
            title="Ensemble Anomaly Score Distribution",
            labels={"ensemble_score": "Anomaly Score", "count": "Events"},
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                       annotation_text=f"Threshold ({threshold})")
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Risk level breakdown
        risk_counts = scores_df["risk_level"].value_counts()
        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Level Distribution",
            color=risk_counts.index,
            color_discrete_map={
                "LOW": "#4caf50", "MEDIUM": "#ffc107",
                "HIGH": "#ff9800", "CRITICAL": "#ff5722", "SEVERE": "#ff1744",
            },
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Timeline
    combined = events_df.copy()
    combined["ensemble_score"] = scores_df["ensemble_score"].values
    combined["is_anomaly_pred"] = scores_df["is_anomaly"].values
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])

    timeline = combined.set_index("timestamp").resample("1h").agg(
        total_events=("event_type", "count"),
        anomalies_detected=("is_anomaly_pred", "sum"),
        avg_score=("ensemble_score", "mean"),
    ).reset_index()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=timeline["timestamp"], y=timeline["total_events"],
               name="Total Events", marker_color="#667eea", opacity=0.6),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=timeline["timestamp"], y=timeline["anomalies_detected"],
                   name="Anomalies", line=dict(color="#ff5722", width=2)),
        secondary_y=True,
    )
    fig.update_layout(
        title="Activity Timeline (Hourly)",
        template="plotly_dark", height=400,
        xaxis_title="Time",
    )
    fig.update_yaxes(title_text="Event Count", secondary_y=False)
    fig.update_yaxes(title_text="Anomalies", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════ TAB 2: ANOMALY EXPLORER ══════════════════
with tab2:
    st.markdown("### 🔍 Anomaly Explorer")

    combined = events_df.copy()
    for col in scores_df.columns:
        combined[col] = scores_df[col].values

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        selected_risk = st.multiselect(
            "Risk Level", ["LOW", "MEDIUM", "HIGH", "CRITICAL", "SEVERE"],
            default=["HIGH", "CRITICAL", "SEVERE"],
        )
    with col_f2:
        selected_events = st.multiselect(
            "Event Type", events_df["event_type"].unique().tolist(),
            default=events_df["event_type"].unique().tolist(),
        )
    with col_f3:
        score_range = st.slider("Score Range", 0.0, 1.0, (0.5, 1.0), step=0.05)

    mask = (
        combined["risk_level"].isin(selected_risk)
        & combined["event_type"].isin(selected_events)
        & combined["ensemble_score"].between(score_range[0], score_range[1])
    )
    filtered = combined[mask].sort_values("ensemble_score", ascending=False)

    st.markdown(f"**{len(filtered):,}** events match filters")

    # Scatter: Autoencoder vs LSTM scores
    fig = px.scatter(
        filtered.head(2000),
        x="autoencoder_score", y="lstm_score",
        color="risk_level",
        size="ensemble_score",
        hover_data=["user_id", "event_type", "ensemble_score"],
        title="Autoencoder vs LSTM Anomaly Scores",
        color_discrete_map={
            "LOW": "#4caf50", "MEDIUM": "#ffc107",
            "HIGH": "#ff9800", "CRITICAL": "#ff5722", "SEVERE": "#ff1744",
        },
    )
    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    display_cols = [
        "timestamp", "user_id", "event_type", "source_host", "dest_ip",
        "bytes_sent", "ensemble_score", "risk_level", "agreement_count",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(
        filtered[display_cols].head(100),
        use_container_width=True, height=400,
    )


# ═══════════════════════════ TAB 3: ALERTS ════════════════════════════
with tab3:
    st.markdown("### ⚠️ Security Alerts")

    if alerts:
        alert_df = pd.DataFrame([a.to_dict() for a in alerts])
        alert_df = alert_df.sort_values("ensemble_score", ascending=False)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Alerts", len(alerts))
        severe = len(alert_df[alert_df["risk_level"].isin(["SEVERE", "CRITICAL"])])
        col2.metric("Critical/Severe", severe)
        col3.metric("Unique Users", alert_df["user_id"].nunique())

        # Alert timeline
        alert_df["timestamp"] = pd.to_datetime(alert_df["timestamp"])
        alert_timeline = alert_df.set_index("timestamp").resample("1h").size().reset_index(name="count")
        fig = px.bar(
            alert_timeline, x="timestamp", y="count",
            title="Alert Volume Over Time",
            color_discrete_sequence=["#ff5722"],
        )
        fig.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig, use_container_width=True)

        # Alert details
        for _, alert_row in alert_df.head(20).iterrows():
            risk = alert_row["risk_level"]
            risk_colors = {
                "SEVERE": "🔴", "CRITICAL": "🟠",
                "HIGH": "🟡", "MEDIUM": "🔵", "LOW": "🟢",
            }
            icon = risk_colors.get(risk, "⚪")
            with st.expander(
                f"{icon} [{alert_row['alert_id']}] {risk} — {alert_row['user_id']} / {alert_row['event_type']} "
                f"(Score: {alert_row['ensemble_score']:.3f})"
            ):
                st.markdown(f"**Description:** {alert_row['description']}")
                st.markdown(f"**Time:** {alert_row['timestamp']}")
                st.markdown(f"**Source:** {alert_row['source_host']} → **Dest:** {alert_row['dest_ip']}")

                model_scores = alert_row["model_scores"]
                score_df = pd.DataFrame({
                    "Model": list(model_scores.keys()),
                    "Score": list(model_scores.values()),
                })
                fig = px.bar(
                    score_df, x="Model", y="Score",
                    color="Score", color_continuous_scale="YlOrRd",
                    title="Individual Model Scores",
                )
                fig.update_layout(template="plotly_dark", height=250)
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alerts generated. Run the pipeline first.")


# ═══════════════════════════ TAB 4: MODEL PERFORMANCE ═════════════════
with tab4:
    st.markdown("### 📈 Model Performance Analysis")

    if eval_metrics:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", f"{eval_metrics['accuracy']:.3f}")
        col2.metric("Precision", f"{eval_metrics['precision']:.3f}")
        col3.metric("Recall", f"{eval_metrics['recall']:.3f}")
        col4.metric("F1 Score", f"{eval_metrics['f1_score']:.3f}")

        st.markdown("---")
        col_left, col_right = st.columns(2)

        with col_left:
            # Confusion matrix
            cm = np.array([
                [eval_metrics["true_negatives"], eval_metrics["false_positives"]],
                [eval_metrics["false_negatives"], eval_metrics["true_positives"]],
            ])
            fig = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Normal", "Anomaly"], y=["Normal", "Anomaly"],
                text_auto=True,
                color_continuous_scale="YlOrRd",
                title="Confusion Matrix",
            )
            fig.update_layout(template="plotly_dark", height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Per-model score distributions for actual anomalies vs normal
            combined = events_df.copy()
            for col in scores_df.columns:
                combined[col] = scores_df[col].values

            actual_anomalies = combined[combined["is_anomaly"] == 1]
            actual_normal = combined[combined["is_anomaly"] == 0]

            models = ["autoencoder_score", "lstm_score", "isolation_forest_score", "behavioral_score"]
            model_labels = ["Autoencoder", "LSTM", "Isolation Forest", "Behavioral"]

            comp_data = []
            for model, label in zip(models, model_labels):
                if model in actual_anomalies.columns:
                    for v in actual_anomalies[model].values:
                        comp_data.append({"Model": label, "Score": v, "Type": "Actual Anomaly"})
                    for v in actual_normal[model].sample(min(500, len(actual_normal)))[model].values:
                        comp_data.append({"Model": label, "Score": v, "Type": "Normal"})

            if comp_data:
                comp_df = pd.DataFrame(comp_data)
                fig = px.box(
                    comp_df, x="Model", y="Score", color="Type",
                    title="Model Scores: Anomaly vs Normal",
                    color_discrete_map={"Actual Anomaly": "#ff5722", "Normal": "#4caf50"},
                )
                fig.update_layout(template="plotly_dark", height=400)
                st.plotly_chart(fig, use_container_width=True)

        # Training metrics
        st.markdown("### Training Metrics")
        if metrics:
            metric_df = pd.DataFrame([{
                "Metric": k.replace("_", " ").title(),
                "Value": v,
            } for k, v in metrics.items()])
            st.dataframe(metric_df, use_container_width=True)
    else:
        st.info("Run the pipeline to see performance metrics.")


# ═══════════════════════════ TAB 5: DEEP DIVE ═════════════════════════
with tab5:
    st.markdown("### 🔬 Deep Dive Analysis")

    if st.session_state.pipeline and st.session_state.pipeline._trained:
        combined = events_df.copy()
        for col in scores_df.columns:
            combined[col] = scores_df[col].values

        # Top risky users
        st.markdown("#### Top Risky Users")
        user_risk = combined.groupby("user_id").agg(
            avg_score=("ensemble_score", "mean"),
            max_score=("ensemble_score", "max"),
            anomaly_count=("is_anomaly", "sum"),
            total_events=("event_type", "count"),
        ).sort_values("avg_score", ascending=False).head(15).reset_index()

        fig = px.bar(
            user_risk, x="user_id", y="avg_score",
            color="anomaly_count",
            color_continuous_scale="YlOrRd",
            title="Top 15 Riskiest Users (by avg anomaly score)",
            hover_data=["max_score", "total_events"],
        )
        fig.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Event type risk heatmap
        st.markdown("#### Event Type Risk Heatmap")
        event_hour_risk = combined.copy()
        event_hour_risk["hour"] = pd.to_datetime(event_hour_risk["timestamp"]).dt.hour
        heatmap_data = event_hour_risk.groupby(["event_type", "hour"])["ensemble_score"].mean().reset_index()
        heatmap_pivot = heatmap_data.pivot(index="event_type", columns="hour", values="ensemble_score").fillna(0)

        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Hour of Day", y="Event Type", color="Avg Score"),
            color_continuous_scale="YlOrRd",
            title="Average Anomaly Score by Event Type and Hour",
            aspect="auto",
        )
        fig.update_layout(template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Latent space visualization (Autoencoder encodings)
        st.markdown("#### Autoencoder Latent Space (t-SNE)")
        pipeline = st.session_state.pipeline

        processed = pipeline.preprocessor.transform(events_df)
        processed = pipeline.temporal_extractor.extract(processed)
        processed = pipeline.graph_extractor.extract(processed)
        X = processed[pipeline.feature_columns].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        encodings = pipeline.autoencoder.get_encodings(X)

        # Use t-SNE for 2D visualization (on a sample for speed)
        sample_size = min(3000, len(encodings))
        indices = np.random.choice(len(encodings), sample_size, replace=False)
        sample_encodings = encodings[indices]
        sample_labels = events_df.iloc[indices]["is_anomaly"].values
        sample_scores = scores_df.iloc[indices]["ensemble_score"].values

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embedded = tsne.fit_transform(sample_encodings)

        tsne_df = pd.DataFrame({
            "TSNE-1": embedded[:, 0],
            "TSNE-2": embedded[:, 1],
            "Anomaly Score": sample_scores,
            "True Label": ["Anomaly" if l == 1 else "Normal" for l in sample_labels],
        })

        fig = px.scatter(
            tsne_df, x="TSNE-1", y="TSNE-2",
            color="True Label",
            size="Anomaly Score",
            color_discrete_map={"Anomaly": "#ff5722", "Normal": "#667eea"},
            title="Autoencoder Latent Space (t-SNE Projection)",
            opacity=0.7,
        )
        fig.update_layout(template="plotly_dark", height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the pipeline to access deep dive analysis.")


# ──────────────────────────── Footer ──────────────────────────────────
st.markdown("---")
st.markdown(
    "<center><small>NexusGuard v1.0 — Adaptive Behavioral Anomaly Detection Engine | "
    "Team MinnalManaf | ACM Nexus 2026</small></center>",
    unsafe_allow_html=True,
)
