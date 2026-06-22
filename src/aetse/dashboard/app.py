import streamlit as st
import duckdb
import pandas as pd
import plotly.express as px
import json

import os

# Fallback to the lightweight DB on Streamlit Cloud if the full DB isn't present
if os.path.exists("data/duckdb/faers.duckdb"):
    DB_PATH = "data/duckdb/faers.duckdb"
else:
    DB_PATH = "data/duckdb/dashboard.duckdb"

st.set_page_config(
    page_title="AET-SE Dashboard",
    page_icon="💊",
    layout="wide"
)

st.title("AET-SE: Adverse Event Triage & Signal Detection Engine")
st.caption("Local-First Pharmacovigilance Signal Detection | LangGraph + Llama 3.1 + FAERS")

@st.cache_resource
def get_conn():
    return duckdb.connect(DB_PATH, read_only=True)

tab1, tab2, tab3 = st.tabs([
    "📊 Signal Explorer",
    "🔍 Agent Trace Viewer", 
    "📈 Metrics"
])

with tab1:
    st.subheader("PRR/ROR Signal Heatmap")
    
    # Drug filter
    drugs = get_conn().execute(
        "SELECT DISTINCT drug FROM prr_signals ORDER BY drug"
    ).df()["drug"].tolist()
    
    selected_drugs = st.multiselect(
        "Select drugs", drugs, 
        default=["ibuprofen", "amlodipine", "lisinopril", "metformin", "rofecoxib"]
    )
    
    min_prr = st.slider("Minimum PRR", 1.0, 20.0, 2.0, 0.5)
    top_n   = st.slider("Top N reactions (by frequency)", 10, 50, 20)
    
    if selected_drugs:
        df = get_conn().execute("""
            SELECT drug, reaction, prr, n_cases, is_signal, masking_warning
            FROM prr_signals
            WHERE drug IN (SELECT unnest(?))
              AND prr >= ?
              AND is_signal = TRUE
            ORDER BY prr DESC
        """, [selected_drugs, min_prr]).df()
        
        if not df.empty:
            # Pivot for heatmap — top N reactions by total n_cases
            top_reactions = (df.groupby("reaction")["n_cases"]
                             .sum().nlargest(top_n).index.tolist())
            pivot = df[df["reaction"].isin(top_reactions)].pivot_table(
                index="reaction", columns="drug", values="prr", aggfunc="max"
            ).fillna(0)
            
            fig = px.imshow(
                pivot,
                color_continuous_scale=["#f8f9fa", "#fecaca", "#ef4444", "#991b1b"],
                title=f"PRR Heatmap — Top {top_n} Reactions",
                labels=dict(x="Drug", y="Adverse Reaction", color="PRR Score"),
                aspect="auto",
                text_auto=".1f"
            )
            fig.update_traces(
                hovertemplate="<b>Drug:</b> %{x}<br><b>Reaction:</b> %{y}<br><b>PRR:</b> %{z:.2f}<extra></extra>"
            )
            fig.update_xaxes(side="top", tickangle=-45)
            fig.update_layout(
                plot_bgcolor="white",
                margin=dict(t=100)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Signal table below heatmap
            st.subheader("Detected Signals Table")
            st.dataframe(
                df[["drug","reaction","prr","n_cases","masking_warning"]]
                .sort_values("prr", ascending=False),
                use_container_width=True
            )
        else:
            st.info("No signals above threshold for selected drugs")

with tab2:
    st.subheader("Pipeline Execution Traces")
    
    # Check if pipeline results exist
    try:
        result_count = get_conn().execute(
            "SELECT COUNT(*) FROM pipeline_results"
        ).fetchone()[0]
    except duckdb.CatalogException:
        result_count = 0
    
    if result_count == 0:
        st.warning(
            "No pipeline results yet. Run the batch pipeline first:\n\n"
            "```bash\nmake run-batch\n```"
        )
    else:
        # Review selector
        reviews = get_conn().execute("""
            SELECT review_id, drug_norm, signal_flag, 
                   extraction_confidence, needs_human_review
            FROM pipeline_results
            ORDER BY processed_at DESC
            LIMIT 200
        """).df()
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Reviews Processed", result_count)
            flag_filter = st.selectbox(
                "Filter by signal flag",
                ["all", "high", "medium", "low", "noise"]
            )
        
        filtered = reviews if flag_filter == "all" else \
                   reviews[reviews["signal_flag"] == flag_filter]
        
        selected_id = st.selectbox(
            "Select review", filtered["review_id"].tolist()
        )
        
        if selected_id:
            try:
                row = get_conn().execute("""
                    SELECT r.review_text, p.*
                    FROM pipeline_results p
                    JOIN drug_reviews r ON p.review_id = r.review_id
                    WHERE p.review_id = ?
                """, [selected_id]).df().iloc[0]
            except duckdb.CatalogException:
                # Fallback if drug_reviews table is missing in the lightweight DB
                row = get_conn().execute("""
                    SELECT 'Review text not available in lightweight cloud deployment.' as review_text, p.*
                    FROM pipeline_results p
                    WHERE p.review_id = ?
                """, [selected_id]).df().iloc[0]
            
            st.subheader("Review Text")
            st.text_area("", row["review_text"], height=150, disabled=True)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Confidence",   f"{row.get('extraction_confidence', 0):.2f}")
            col2.metric("Signal Flag",  row.get("signal_flag") or "none")
            col3.metric("Severity",     row.get("severity") or "unknown")
            col4.metric("Human Review", "Yes" if row.get("needs_human_review") else "No")
            
            st.subheader("Agent Execution Trace")
            trace = json.loads(row["agent_trace"] or "[]")
            
            # Color map for trace steps
            colors = {
                "EXTRACTION":    "#4CAF50",
                "EXTRACTION_RETRY": "#F44336",
                "VALIDATION":    "#2196F3",
                "MAPPING":       "#FF9800",
                "SIGNAL_CHECK":  "#9C27B0",
                "HUMAN_FLAG":    "#795548"
            }
            
            for i, step in enumerate(trace):
                agent = step.split(":")[0]
                color = colors.get(agent, "#666666")
                st.markdown(
                    f'<div style="display:flex;align-items:center;'
                    f'margin:4px 0;padding:8px;border-radius:6px;'
                    f'background:#f8f9fa;border-left:4px solid {color}">'
                    f'<span style="background:{color};color:white;'
                    f'padding:2px 8px;border-radius:10px;font-size:11px;'
                    f'margin-right:10px">Step {i+1}</span>'
                    f'<code style="font-size:12px">{step}</code></div>',
                    unsafe_allow_html=True
                )
            
            # Signals found
            signals = json.loads(row["prr_signals"] or "[]")
            if signals:
                st.subheader("PRR Signals Detected")
                st.dataframe(pd.DataFrame(signals), use_container_width=True)
            
            # Latency breakdown
            st.subheader("Node Latency")
            latency_data = {
                "Node": ["extract", "map_terms", "signal_check"],
                "Latency (ms)": [
                    row.get("extract_latency_ms") or 0,
                    row.get("map_terms_latency_ms") or 0,
                    row.get("signal_check_latency_ms") or 0
                ]
            }
            fig = px.bar(
                pd.DataFrame(latency_data),
                x="Node", y="Latency (ms)",
                color="Node", title="Processing Time per Node"
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Pipeline Performance Metrics")
    
    # PRR/FAERS summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    try:
        faers_count = get_conn().execute(
            "SELECT COUNT(*) FROM faers_cases"
        ).fetchone()[0]
    except duckdb.CatalogException:
        faers_count = 776057  # Fallback static count
        
    try:
        signal_count = get_conn().execute(
            "SELECT COUNT(*) FROM prr_signals WHERE is_signal = TRUE"
        ).fetchone()[0]
    except duckdb.CatalogException:
        signal_count = 0
        
    try:
        review_count = get_conn().execute(
            "SELECT COUNT(*) FROM drug_reviews WHERE has_ae_mention = TRUE"
        ).fetchone()[0]
    except duckdb.CatalogException:
        review_count = 2159  # Fallback static count
    try:
        processed_count = get_conn().execute(
            "SELECT COUNT(*) FROM pipeline_results"
        ).fetchone()[0]
    except duckdb.CatalogException:
        processed_count = 0
    
    col1.metric("FAERS Cases", f"{faers_count:,}")
    col2.metric("Signals Detected", f"{signal_count:,}")
    col3.metric("AE Reviews", f"{review_count:,}")
    col4.metric("Reviews Processed", f"{processed_count:,}")
    
    # Latency breakdown from pipeline_results
    if processed_count > 0:
        st.subheader("Node Latency Distribution")
        latency_df = get_conn().execute("""
            SELECT 
                AVG(extract_latency_ms)      as avg_extract,
                AVG(map_terms_latency_ms)    as avg_map_terms,
                AVG(signal_check_latency_ms) as avg_signal_check,
                MAX(extract_latency_ms)      as max_extract,
                MAX(map_terms_latency_ms)    as max_map_terms,
                MAX(signal_check_latency_ms) as max_signal_check
            FROM pipeline_results
            WHERE extract_latency_ms IS NOT NULL
        """).df()
        
        nodes    = ["extract", "map_terms", "signal_check"]
        avg_vals = [
            latency_df["avg_extract"].iloc[0],
            latency_df["avg_map_terms"].iloc[0],
            latency_df["avg_signal_check"].iloc[0]
        ]
        max_vals = [
            latency_df["max_extract"].iloc[0],
            latency_df["max_map_terms"].iloc[0],
            latency_df["max_signal_check"].iloc[0]
        ]
        
        import plotly.graph_objects as go
        fig = go.Figure(data=[
            go.Bar(name="Avg (ms)", x=nodes, y=avg_vals, marker_color="#2196F3"),
            go.Bar(name="Max (ms)", x=nodes, y=max_vals, marker_color="#F44336"),
        ])
        fig.update_layout(
            barmode="group",
            title="Pipeline Node Latency (ms) — Avg vs Max",
            yaxis_title="Latency (ms)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confidence score distribution
        st.subheader("Extraction Confidence Distribution")
        conf_df = get_conn().execute("""
            SELECT extraction_confidence FROM pipeline_results
        """).df()
        fig2 = px.histogram(
            conf_df, x="extraction_confidence",
            nbins=20, title="Confidence Score Distribution",
            labels={"extraction_confidence": "Confidence Score"},
            color_discrete_sequence=["#4CAF50"]
        )
        fig2.add_vline(x=0.75, line_dash="dash", 
                       line_color="red",
                       annotation_text="Routing threshold (0.75)")
        st.plotly_chart(fig2, use_container_width=True)

    # Positive control validation
    st.subheader("Positive Control Signal Validation")
    controls = [
        ("ibuprofen",  "Gastrointestinal haemorrhage", 2.0),
        ("rofecoxib",  "Myocardial infarction",        2.0),
        ("metformin",  "Lactic acidosis",              2.0),
        ("amlodipine", "Oedema peripheral",            2.0),
        ("lisinopril", "Cough",                        2.0),
    ]
    rows = []
    for drug, reaction, threshold in controls:
        result = get_conn().execute("""
            SELECT prr, n_cases, is_signal FROM prr_signals
            WHERE LOWER(drug) = LOWER(?)
              AND LOWER(reaction) = LOWER(?)
        """, [drug, reaction]).fetchone()
        rows.append({
            "Drug":      drug,
            "Reaction":  reaction,
            "PRR":       result[0] if result else None,
            "N Cases":   result[1] if result else None,
            "Signal":    "✅ PASS" if result and result[2] else "❌ FAIL",
            "Threshold": f"PRR > {threshold}"
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    
    # Signal flag distribution
    if processed_count > 0:
        st.subheader("Signal Flag Distribution (Processed Reviews)")
        flag_dist = get_conn().execute("""
            SELECT signal_flag, COUNT(*) as count
            FROM pipeline_results
            GROUP BY signal_flag
            ORDER BY count DESC
        """).df()
        fig = px.pie(
            flag_dist, values="count", names="signal_flag",
            title="Pipeline Output Distribution",
            color_discrete_map={
                "high":"#d32f2f","medium":"#f57c00",
                "low":"#388e3c","noise":"#9e9e9e"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
