import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Social Pulse Intelligence",
    page_icon="⚡",
    layout="wide"
)

# Custom Clean Dark Theme Styling
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    
    /* Input Area Styling */
    .stTextArea textarea {
        background-color: #161b22;
        color: #f0f6fc;
        border: 1px solid #30363d;
        border-radius: 8px;
        font-size: 16px;
    }
    
    /* Result Cards Styling */
    .kpi-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-title { color: #8b949e; font-size: 14px; font-weight: 600; text-transform: uppercase; }
    .kpi-value { font-size: 28px; font-weight: 800; margin-top: 5px; }
    
    /* Aspect Badges */
    .aspect-badge {
        display: inline-block;
        background-color: #1f242c;
        border: 1px solid #388bfd;
        color: #58a6ff;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 5px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 24px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #21262d !important;
        color: #58a6ff !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

API_URL = "http://127.0.0.1:8000/predict"

st.title("⚡ Social Pulse — AI Intelligence Dashboard")
st.caption("Real-Time Sentiment Analysis, Aspect Extraction, and Emotional Intelligence Engine")

# Sidebar Status
st.sidebar.header("🌐 Engine Status")
try:
    health = requests.get("http://127.0.0.1:8000/health", timeout=2)
    if health.status_code == 200:
        st.sidebar.success("🟢 API Connected")
    else:
        st.sidebar.warning("🟡 API Warning")
except Exception:
    st.sidebar.error("🔴 API Offline! Start `uvicorn app:app`")

tab1, tab2 = st.tabs(["🎯 Real-Time Deep Analysis", "📊 Batch Dataset Analytics"])

# ==========================================
# TAB 1: SINGLE POST ANALYSIS (Full-Width Layout)
# ==========================================
with tab1:
    st.subheader("1. Input Social Media Post")
    
    # Full-Width Input Section
    user_input = st.text_area(
        label="Type or paste post, tweet, or product review below:",
        value="I love the new mobile app UI design, but the battery drain issue is completely unacceptable! @support #tech",
        height=100,
        placeholder="Type something here..."
    )
    
    analyze_btn = st.button("🚀 Run Deep Analysis", use_container_width=True, type="primary")

    # Full-Width Output Section Below
    if analyze_btn and user_input.strip():
        st.markdown("---")
        st.subheader("2. Intelligence Output & Breakdown")

        try:
            res = requests.post(API_URL, json={"text": user_input})
            if res.status_code == 200:
                data = res.json()
                label = data["sentiment_label"].capitalize()
                conf = round(data["confidence"] * 100, 1)

                # Row 1: High-Level KPI Summary Cards
                kpi1, kpi2, kpi3 = st.columns([1.5, 1.5, 3])

                with kpi1:
                    color_map = {"Positive": "#10B981", "Negative": "#EF4444", "Neutral": "#3B82F6"}
                    card_color = color_map.get(label, "#3B82F6")
                    st.markdown(f"""
                    <div class="kpi-card" style="border-left: 5px solid {card_color};">
                        <div class="kpi-title">Overall Sentiment</div>
                        <div class="kpi-value" style="color: {card_color};">{label}</div>
                    </div>
                    """, unsafe_allow_html=True)

                with kpi2:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-title">Confidence Score</div>
                        <div class="kpi-value" style="color: #f0f6fc;">{conf}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                with kpi3:
                    aspects_html = "".join([f'<span class="aspect-badge">🏷️ {a}</span>' for a in data["aspects"]])
                    st.markdown(f"""
                    <div class="kpi-card" style="text-align: left;">
                        <div class="kpi-title">Detected Topics / Aspects</div>
                        <div style="margin-top: 10px;">{aspects_html}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Row 2: Visualizations Side-by-Side
                chart_col1, chart_col2 = st.columns(2)

                # Visual 1: Sentiment Probability Distribution
                with chart_col1:
                    st.markdown("#### 📊 Sentiment Probability Breakdown")
                    probs = data["probabilities"]
                    df_probs = pd.DataFrame({
                        "Sentiment": [k.capitalize() for k in probs.keys()],
                        "Score (%)": [v * 100 for v in probs.values()]
                    })
                    
                    fig_bar = px.bar(
                        df_probs, 
                        x="Score (%)", 
                        y="Sentiment", 
                        orientation="h",
                        color="Sentiment",
                        color_discrete_map={"Positive": "#10B981", "Neutral": "#3B82F6", "Negative": "#EF4444"},
                        text_auto=".1f"
                    )
                    fig_bar.update_layout(
                        showlegend=False, 
                        height=280, 
                        margin=dict(l=0, r=0, t=20, b=0),
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#ffffff"),
                        xaxis=dict(showgrid=False),
                        yaxis=dict(showgrid=False)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)

                # Visual 2: Multi-Dimensional Emotional Profile (Radar Chart)
                with chart_col2:
                    st.markdown("#### 🎭 Emotional Intensity Profile")
                    emotions = data["emotions"]
                    categories = list(emotions.keys())
                    values = list(emotions.values())

                    fig_radar = go.Figure()
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values, theta=categories, fill='toself',
                        line_color='#8B5CF6', fillcolor='rgba(139, 92, 246, 0.3)'
                    ))
                    fig_radar.update_layout(
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
                            bgcolor="rgba(22, 27, 34, 0.5)"
                        ),
                        showlegend=False, 
                        height=280, 
                        margin=dict(l=30, r=30, t=20, b=20),
                        paper_bgcolor="rgba(0,0,0,0)", 
                        font=dict(color="#ffffff")
                    )
                    st.plotly_chart(fig_radar, use_container_width=True)

            else:
                st.error(f"Error from API Engine: {res.text}")
        except Exception as e:
            st.error(f"Could not reach API service: {e}")

# ==========================================
# TAB 2: BATCH DATASET ANALYTICS
# ==========================================
with tab2:
    st.subheader("Upload Social Media Dataset (CSV)")
    file = st.file_uploader("Choose a CSV file containing social media posts/comments", type=["csv"])

    if file is not None:
        df = pd.read_csv(file)
        st.write("Dataset Preview:", df.head(3))
        
        target_col = st.selectbox("Select Text Column:", df.columns)

        if st.button("⚡ Process Dataset in Bulk"):
            results_label = []
            results_aspect = []
            progress = st.progress(0)
            total = len(df)

            for i, text in enumerate(df[target_col]):
                try:
                    r = requests.post(API_URL, json={"text": str(text)})
                    if r.status_code == 200:
                        out = r.json()
                        results_label.append(out["sentiment_label"])
                        results_aspect.append(out["aspects"][0])
                    else:
                        results_label.append("error")
                        results_aspect.append("General")
                except Exception:
                    results_label.append("failed")
                    results_aspect.append("General")

                progress.progress((i + 1) / total)

            df["Sentiment"] = results_label
            df["Aspect"] = results_aspect

            st.markdown("---")
            
            # High-Level Metric Cards
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Posts Analyzed", len(df))
            m2.metric("Positive Comments", (df["Sentiment"] == "positive").sum())
            m3.metric("Neutral Comments", (df["Sentiment"] == "neutral").sum())
            m4.metric("Negative Comments", (df["Sentiment"] == "negative").sum())

            # Multi-Dimensional Sunburst Chart
            st.markdown("#### Aspect-Based Sentiment Hierarchy")
            df_grouped = df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
            
            fig_sunburst = px.sunburst(
                df_grouped,
                path=["Aspect", "Sentiment"],
                values="Count",
                color="Sentiment",
                color_discrete_map={"positive": "#10B981", "neutral": "#3B82F6", "negative": "#EF4444"}
            )
            fig_sunburst.update_layout(height=450, paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#ffffff"))
            st.plotly_chart(fig_sunburst, use_container_width=True)

            # Annotated Table & Download
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "📥 Download Annotated Dataset",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="social_sentiment_results.csv",
                mime="text/csv"
            )