"""
gui/app.py
----------
Streamlit dashboard for NEXUS 2026.

Run with:
  streamlit run gui/app.py
"""

import sys
import json
import random
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path fix so src.app imports work when run from any cwd ────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.app.ingestion  import load_log
from src.app.processing import build_features
from src.app.analysis   import run_analysis, SEVERITY_ORDER, Finding
from data.generate_logs import build_dataset, write_csv

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NEXUS 2026",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour map ────────────────────────────────────────────────────────────────

SEV_COLOURS = {
    "CRITICAL": "#e74c3c",
    "HIGH":     "#e67e22",
    "MEDIUM":   "#f1c40f",
    "LOW":      "#3498db",
}

SEV_ICONS = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🔵",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def severity_badge(sev: str) -> str:
    colour = SEV_COLOURS.get(sev, "#aaa")
    icon   = SEV_ICONS.get(sev, "")
    return f'<span style="background:{colour};color:#fff;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold">{icon} {sev}</span>'


def findings_to_df(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Severity":    f.severity,
            "Entity Type": f.entity_type.upper(),
            "Entity":      f.entity_value,
            "Category":    f.category,
            "Layer":       f.layer,
            "Description": f.description,
            "Evidence":    json.dumps(f.evidence),
        }
        for f in findings
    ])


@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes: bytes) -> tuple:
    """Cache the full pipeline result keyed on file content."""
    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="wb") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    df         = load_log(tmp_path)
    ip_f, u_f  = build_features(df)
    findings   = run_analysis(ip_f, u_f)
    return df, ip_f, u_f, findings


def generate_log_bytes(n_rows: int, seed: int) -> bytes:
    random.seed(seed)
    base    = datetime(2026, 3, 27)
    dataset = build_dataset(n_rows, base)

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False, mode="w", newline="") as tmp:
        import csv
        writer = csv.DictWriter(tmp, fieldnames=["timestamp", "source_ip", "username", "action"])
        writer.writeheader()
        writer.writerows(dataset)
        tmp_path = tmp.name

    return Path(tmp_path).read_bytes()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=64)
    st.title("NEXUS 2026")
    st.caption("Activity Pattern Analysis & Threat Detection")
    st.divider()

    st.subheader("📂 Data Source")

    data_source = st.radio(
        "Choose input",
        ["Upload a log file", "Generate synthetic data"],
        label_visibility="collapsed",
    )

    file_bytes: bytes | None = None

    if data_source == "Upload a log file":
        uploaded = st.file_uploader(
            "Upload CSV log",
            type=["log", "csv", "txt"],
            help="Columns: timestamp, source_ip, username, action",
        )
        if uploaded:
            file_bytes = uploaded.read()

    else:
        n_rows = st.slider("Normal events", 100, 2000, 500, step=100)
        seed   = st.number_input("Random seed", value=42, step=1)
        if st.button("⚡ Generate & Analyse", use_container_width=True, type="primary"):
            with st.spinner("Generating synthetic log…"):
                file_bytes = generate_log_bytes(n_rows, int(seed))
            st.session_state["file_bytes"] = file_bytes

    # Persist between reruns
    if file_bytes:
        st.session_state["file_bytes"] = file_bytes

    st.divider()
    st.subheader("🔧 Filters")
    min_sev = st.selectbox(
        "Minimum severity",
        ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        index=0,
    )
    filter_layer = st.multiselect(
        "Detection layers",
        ["rules", "zscore", "isolation_forest"],
        default=["rules", "zscore", "isolation_forest"],
    )
    filter_entity = st.multiselect(
        "Entity type",
        ["IP", "USER"],
        default=["IP", "USER"],
    )

# ── Main content ──────────────────────────────────────────────────────────────

if "file_bytes" not in st.session_state:
    # Landing screen
    st.markdown(
        """
        <div style="text-align:center;padding:80px 0 40px">
            <img src="https://img.icons8.com/fluency/128/cyber-security.png"/>
            <h1 style="margin-top:16px">NEXUS 2026</h1>
            <p style="font-size:1.1em;color:#888">
                Upload an activity log or generate synthetic data using the sidebar to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Rule-based detection**\nHard thresholds for brute force, credential stuffing, impossible travel, and after-hours access.")
    with col2:
        st.info("**Z-score baseline**\nFlags features that deviate ≥ 2.5σ from the population — adapts as data grows.")
    with col3:
        st.info("**Isolation Forest (ML)**\nUnsupervised learning that scores multi-dimensional outliers without labelled data.")
    st.stop()


# ── Run pipeline ──────────────────────────────────────────────────────────────

with st.spinner("Running analysis pipeline…"):
    df, ip_feats, user_feats, all_findings = run_pipeline(st.session_state["file_bytes"])

# Apply sidebar filters
min_rank = SEVERITY_ORDER[min_sev]
findings = [
    f for f in all_findings
    if SEVERITY_ORDER[f.severity] <= min_rank
    and f.layer in filter_layer
    and f.entity_type.upper() in filter_entity
]

# ── Header KPIs ───────────────────────────────────────────────────────────────

st.markdown("## 🔍 Threat Intelligence Report")
st.caption(f"Analysed **{len(df):,}** events · {len(ip_feats)} source IPs · {len(user_feats)} users · {len(findings)} findings shown")

counts = {s: sum(1 for f in findings if f.severity == s) for s in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]}

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Findings", len(findings))
k2.metric("🔴 CRITICAL", counts["CRITICAL"], delta=None)
k3.metric("🟠 HIGH",     counts["HIGH"],     delta=None)
k4.metric("🟡 MEDIUM",   counts["MEDIUM"],   delta=None)
k5.metric("🔵 LOW",      counts["LOW"],      delta=None)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_findings, tab_charts, tab_ip, tab_user, tab_events = st.tabs([
    "🚨 Findings",
    "📊 Charts",
    "🌐 IP Features",
    "👤 User Features",
    "📋 Raw Events",
])


# ────────────────────────── Tab 1: Findings ──────────────────────────────────

with tab_findings:
    if not findings:
        st.success("✅ No findings match the current filters.")
    else:
        fdf = findings_to_df(findings)

        # Colour-code severity column via HTML
        def style_severity(val):
            c = SEV_COLOURS.get(val, "#ccc")
            return f"background-color:{c};color:white;font-weight:bold;border-radius:4px;padding:2px 6px"

        st.dataframe(
            fdf.style.map(style_severity, subset=["Severity"]),
            use_container_width=True,
            height=500,
            hide_index=True,
        )

        # JSON export
        records = [
            {
                "entity_type":  f.entity_type,
                "entity_value": f.entity_value,
                "severity":     f.severity,
                "category":     f.category,
                "description":  f.description,
                "layer":        f.layer,
                "evidence":     f.evidence,
            }
            for f in findings
        ]
        report_json = json.dumps(
            {"generated_at": datetime.utcnow().isoformat() + "Z",
             "total_findings": len(records), "findings": records},
            indent=2,
        )
        st.download_button(
            "⬇️ Download JSON Report",
            data=report_json,
            file_name="nexus_report.json",
            mime="application/json",
        )


# ────────────────────────── Tab 2: Charts ────────────────────────────────────

with tab_charts:
    if not findings:
        st.info("No findings to chart with the current filters.")
    else:
        fdf = findings_to_df(findings)

        c1, c2 = st.columns(2)

        with c1:
            # Severity distribution
            sev_counts = (
                fdf["Severity"]
                .value_counts()
                .reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"])
                .dropna()
                .reset_index()
            )
            sev_counts.columns = ["Severity", "Count"]
            fig = px.bar(
                sev_counts, x="Severity", y="Count",
                color="Severity",
                color_discrete_map=SEV_COLOURS,
                title="Findings by Severity",
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Category breakdown
            cat_counts = fdf["Category"].value_counts().head(10).reset_index()
            cat_counts.columns = ["Category", "Count"]
            fig2 = px.bar(
                cat_counts, x="Count", y="Category",
                orientation="h",
                title="Top Threat Categories",
                color="Count",
                color_continuous_scale="Reds",
            )
            fig2.update_layout(coloraxis_showscale=False, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)

        with c3:
            # Layer breakdown
            layer_counts = fdf["Layer"].value_counts().reset_index()
            layer_counts.columns = ["Layer", "Count"]
            fig3 = px.pie(
                layer_counts, names="Layer", values="Count",
                title="Detections by Layer",
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c4:
            # IP vs User split
            entity_counts = fdf["Entity Type"].value_counts().reset_index()
            entity_counts.columns = ["Entity Type", "Count"]
            fig4 = px.pie(
                entity_counts, names="Entity Type", values="Count",
                title="Findings: IP vs User",
                color_discrete_sequence=["#3498db", "#e74c3c"],
                hole=0.4,
            )
            st.plotly_chart(fig4, use_container_width=True)

        # Event timeline
        st.subheader("Event Timeline")
        timeline_df = df.copy()
        timeline_df["hour"] = timeline_df["timestamp"].dt.hour
        hourly = timeline_df.groupby(["hour", "action"]).size().reset_index(name="count")
        fig5 = px.bar(
            hourly, x="hour", y="count", color="action",
            title="Events by Hour of Day",
            labels={"hour": "Hour (UTC)", "count": "Event Count"},
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        fig5.add_vrect(x0=0, x1=8,  fillcolor="red",  opacity=0.06, line_width=0, annotation_text="After hours")
        fig5.add_vrect(x0=18, x1=23, fillcolor="red",  opacity=0.06, line_width=0)
        st.plotly_chart(fig5, use_container_width=True)


# ────────────────────────── Tab 3: IP Features ───────────────────────────────

with tab_ip:
    st.subheader("Per-IP Behavioural Features")

    # Highlight flagged IPs
    flagged_ips = {f.entity_value for f in findings if f.entity_type == "ip"}
    ip_display  = ip_feats.copy()
    ip_display["⚠ Flagged"] = ip_display["source_ip"].isin(flagged_ips).map({True: "YES", False: ""})

    # Sort flagged to top
    ip_display = ip_display.sort_values("⚠ Flagged", ascending=False)

    def highlight_flagged(row):
        if row["⚠ Flagged"] == "YES":
            return ["background-color:#fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        ip_display.style.apply(highlight_flagged, axis=1),
        use_container_width=True,
        height=420,
        hide_index=True,
    )

    # Top IPs by failure count
    top_ips = ip_feats.nlargest(10, "failure_count")
    fig = px.bar(
        top_ips, x="source_ip", y="failure_count",
        color="failure_rate",
        color_continuous_scale="Reds",
        title="Top 10 IPs by Login Failure Count",
        labels={"source_ip": "Source IP", "failure_count": "Failures", "failure_rate": "Failure Rate"},
    )
    fig.update_xaxes(tickangle=30)
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────── Tab 4: User Features ─────────────────────────────

with tab_user:
    st.subheader("Per-User Behavioural Features")

    flagged_users = {f.entity_value for f in findings if f.entity_type == "user"}
    user_display  = user_feats.copy()
    user_display["⚠ Flagged"] = user_display["username"].isin(flagged_users).map({True: "YES", False: ""})
    user_display  = user_display.sort_values("⚠ Flagged", ascending=False)

    def highlight_flagged_user(row):
        if row["⚠ Flagged"] == "YES":
            return ["background-color:#fff3cd"] * len(row)
        return [""] * len(row)

    st.dataframe(
        user_display.style.apply(highlight_flagged_user, axis=1),
        use_container_width=True,
        height=300,
        hide_index=True,
    )

    # Unique IPs per user radar
    fig = px.bar(
        user_feats.sort_values("unique_ips", ascending=False),
        x="username", y=["unique_ips", "sudo_count", "after_hours_count"],
        barmode="group",
        title="User Risk Indicators",
        labels={"value": "Count", "variable": "Indicator"},
        color_discrete_sequence=["#3498db", "#e74c3c", "#f39c12"],
    )
    st.plotly_chart(fig, use_container_width=True)


# ────────────────────────── Tab 5: Raw Events ────────────────────────────────

with tab_events:
    st.subheader("Raw Event Log")

    action_filter = st.multiselect(
        "Filter by action",
        options=sorted(df["action"].unique()),
        default=sorted(df["action"].unique()),
    )

    filtered_df = df[df["action"].isin(action_filter)] if action_filter else df
    st.dataframe(filtered_df, use_container_width=True, height=480, hide_index=True)
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} events")
