import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
import uuid
from openai import OpenAI # Ensure openai library is installed: pip install openai

# Set page configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f7;
        color: #1d1d1f;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #007aff;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 8px 16px;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .metric-title {
        font-size: 18px;
        font-weight: bold;
        color: #1d1d1f;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #34c759;
    }
    .metric-trend-positive {
        color: #34c759;
        font-size: 14px;
    }
    .metric-trend-negative {
        color: #ff3b30;
        font-size: 14px;
    }
    .stRadio > div {
        display: flex;
        gap: 10px;
    }
    .stRadio > div > label {
        background-color: #ffffff;
        padding: 8px 16px;
        border-radius: 20px;
        border: 1px solid #e5e5ea;
        font-size: 14px;
    }
    .stRadio > div > label:hover {
        background-color: #f5f5f7;
        cursor: pointer;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- NVIDIA API Client Initialization ---
# WARNING: Hardcoding API keys is a security risk. Use secrets management in production.
SYSTEM_PROMPT_VIRA = """
Anda adalah VIRA, seorang konsultan virtual untuk Bank BCA.
Tugas utama Anda adalah menganalisis data dasbor yang disediakan dan memberikan wawasan, ringkasan, serta saran yang relevan.
Fokuslah pada metrik seperti skor kesehatan, tren, sentimen pelanggan, niat panggilan, dan volume panggilan.
Selalu dasarkan jawaban Anda pada data yang diberikan dalam `dashboard_state`.
Gunakan bahasa Indonesia yang sopan dan mudah dimengerti.
Jika ada pertanyaan yang tidak dapat dijawab dari data dasbor, sampaikan dengan sopan bahwa informasi tersebut tidak tersedia dalam tampilan dasbor saat ini atau minta pengguna untuk memberikan detail lebih lanjut.
Berikan analisis yang ringkas namun mendalam.
Jika ada pertanyaan yang diluar konteks analisis anda, sampaikan bahwa itu diluar kapabilitas anda untuk menjelaskannya

PENTING:
Sebelum memberikan jawaban akhir kepada pengguna, Anda BOLEH melakukan analisis internal atau "berpikir".
Jika Anda melakukan proses berpikir internal, *JANGAN* tuliskan pemikiran tersebut.
Jika tidak ada proses berpikir khusus atau analisis internal yang perlu dituliskan, langsung berikan jawaban
"""

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "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" # Replace with your actual API key or use secrets
)

def generate_llm_response(user_prompt: str, dashboard_state: dict, system_prompt: str):
    """
    Generates a response from the LLM based on the user prompt, dashboard state, and system prompt.
    Streams the response.
    """
    # Format dashboard state untuk LLM agar lebih mudah dibaca
    dashboard_summary_for_llm = f"""
Ringkasan tampilan dasbor saat ini berdasarkan filter yang dipilih:
- Periode Waktu Terpilih untuk Skor Kesehatan: {dashboard_state.get('time_period_label', 'N/A')}
- Skor Kesehatan Pelanggan: {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')} - {dashboard_state.get('trend_label', 'N/A')})

Ringkasan Grafik Langsung (perkiraan berdasarkan filter saat ini):
- Distribusi Sentimen: Positif: {dashboard_state.get('sentiment_summary', {}).get('Positive', 'N/A')}, Netral: {dashboard_state.get('sentiment_summary', {}).get('Neutral', 'N/A')}, Negatif: {dashboard_state.get('sentiment_summary', {}).get('Negative', 'N/A')}.
- Distribusi Niat: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('intent_summary', {}).items()]) if dashboard_state.get('intent_summary') else 'N/A'}.
- Tren Volume: {dashboard_state.get('volume_summary', 'N/A')}.

Informasi Dasbor Umum (ini adalah contoh, peringatan/hotspot spesifik dapat bervariasi dan harus diperiksa pada kartunya masing-masing):
- Peringatan Kritis: Dapat menyoroti masalah seperti "Lonjakan Mendadak dalam Sentimen Negatif" atau "Risiko Churn Tinggi".
- Hotspot Prediktif: Bisa menunjuk ke "Kebingungan Kebijakan" atau "Masalah UI".
- Tema Pelanggan Teratas (Positif): Contohnya "Layanan Pelanggan Cepat", "Mobile Banking Mudah".
- Tema Pelanggan Teratas (Negatif): Contohnya "Masalah Teknis Aplikasi", "Waktu Tunggu Lama".
- Radar Peluang: Mengidentifikasi area seperti "Fitur yang Menyenangkan", "Peluang Cross-Sell", "Keunggulan Layanan".
"""

    constructed_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{dashboard_summary_for_llm}\n\nPertanyaan Pengguna: \"{user_prompt}\""}
    ]

    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", # As per user's original code
            messages=constructed_messages, # Use the constructed messages
            temperature=1.00, # As per user's original code
            top_p=0.01,       # As per user's original code
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_message = f"Maaf, terjadi kesalahan saat menghubungi layanan AI: {str(e)}. Silakan coba lagi nanti atau periksa konsol."
        # Log error ke konsol juga untuk debugging
        print(f"LLM API Error: {e}")
        # Di Streamlit, error ditampilkan di UI, jadi yield pesan error agar placeholder bisa menampilkannya
        yield error_message

# Generate health score data
def generate_health_score_data():
    return {
        "today": {
            "labels": ["9 AM", "11 AM", "1 PM", "3 PM", "5 PM", "7 PM", "9 PM"],
            "values": [78, 76, 80, 79, 81, 83, 84],
            "score": 84,
            "trend": "+2.5%",
            "trend_positive": True,
            "trend_label": "vs. yesterday",
        },
        "week": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "values": [79, 78, 80, 81, 83, 84, 85],
            "score": 85,
            "trend": "+1.8%",
            "trend_positive": True,
            "trend_label": "vs. last week",
        },
        "month": {
            "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
            "values": [79, 80, 81, 82],
            "score": 82,
            "trend": "+1.5%",
            "trend_positive": True,
            "trend_label": "vs. last month",
        },
        "quarter": {
            "labels": ["Jan", "Feb", "Mar"],
            "values": [76, 79, 83],
            "score": 83,
            "trend": "+3.2%",
            "trend_positive": True,
            "trend_label": "vs. last quarter",
        },
        "year": {
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "values": [75, 77, 80, 84],
            "score": 84,
            "trend": "+4.1%",
            "trend_positive": True,
            "trend_label": "vs. last year",
        },
        "all": {
            "labels": ["2019", "2020", "2021", "2022", "2023", "2024"],
            "values": [73, 71, 75, 78, 80, 83],
            "score": 83,
            "trend": "+10.4%",
            "trend_positive": True,
            "trend_label": "over 5 years",
        },
    }

# Sidebar
with st.sidebar:
    st.title("VOCAL")
    st.markdown("---")

    st.header("Menu")
    page = st.selectbox("Navigate", ["Dashboard", "Analytics", "Feedback", "Alerts", "Reports"], key="menu_nav")

    st.header("Customer Insights")
    st.selectbox("Insights", ["Sentiment Analysis", "Journey Mapping", "Satisfaction Scores", "Theme Analysis"], key="insights_nav")

    st.header("Operations")
    st.selectbox("Operations", ["Real-time Monitoring", "Predictive Analytics", "Performance Metrics", "Action Items"], key="ops_nav")

    st.header("Configuration")
    st.selectbox("Config", ["Settings", "User Management", "Security", "Help & Support"], key="config_nav")

    st.markdown("---")
    st.markdown("**Sebastian**")
    st.markdown("CX Manager")

# Main content
if page == "Dashboard":
    st.title("Customer Experience Health")
    st.markdown("Real-time Insights & Performance Overview")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        time_period = st.selectbox(
            "TIME",
            ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
            index=3, # Default to "This Month"
            key="time_filter"
        )
    with col2:
        products = st.multiselect(
            "PRODUCT",
            ["All Products", "myBCA", "BCA Mobile", "KPR", "KKB", "KSM", "Investasi", "Asuransi", "KMK", "Kartu Kredit", "EDC & QRIS", "Poket Valas"],
            default=["All Products"],
            key="product_filter"
        )
    with col3:
        channels = st.multiselect(
            "CHANNEL",
            ["All Channels", "Social Media", "Call Center", "WhatsApp", "Webchat", "VIRA", "E-mail", "Survey Gallup", "Survey BSQ", "Survey CX"],
            default=["All Channels"],
            key="channel_filter"
        )

    # Filter logic
    time_period_map = {
        "All Periods": "all",
        "Today": "today",
        "This Week": "week",
        "This Month": "month",
        "This Quarter": "quarter",
        "This Year": "year"
    }
    selected_time_key = time_period_map.get(time_period, "month") # e.g. "month"
    product_filter_active = ["all"] if "All Products" in products else [p.lower().replace(" ", "_") for p in products]
    channel_filter_active = ["all"] if "All Channels" in channels else [c.lower().replace(" ", "_") for c in channels]

    # Multipliers for filtering (example effect)
    product_multiplier = 1.0 if "all" in product_filter_active else 0.8
    channel_multiplier = 1.0 if "all" in channel_filter_active else 0.9
    if "social_media" in channel_filter_active and "all" not in channel_filter_active:
        channel_multiplier = 0.6

    # Health score data
    health_score_data_source = generate_health_score_data()
    current_health_data = health_score_data_source.get(selected_time_key, health_score_data_source["month"]).copy() # Use .copy() to avoid modifying the source
    current_health_data['time_period_label'] = time_period # Add display name of time period, e.g. "This Month"


    # Dashboard widgets
    st.markdown("## Dashboard Widgets")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Customer Health Score")
        health_view = st.radio("View", ["Real-time", "Daily Trend", "Comparison"], horizontal=True, key="health_view")

        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f'<div class="metric-value">{current_health_data["score"]}%</div>', unsafe_allow_html=True)
        with score_col2:
            trend_icon = "↑" if current_health_data["trend_positive"] else "↓"
            trend_class = "metric-trend-positive" if current_health_data["trend_positive"] else "metric-trend-negative"
            st.markdown(f'<div class="{trend_class}">{trend_icon} {current_health_data["trend"]} {current_health_data["trend_label"]}</div>', unsafe_allow_html=True)

        fig_health = go.Figure()
        fig_health.add_trace(go.Scatter(
            x=current_health_data["labels"],
            y=current_health_data["values"],
            mode='lines',
            fill='tozeroy',
            fillcolor='rgba(52,199,89,0.18)',
            line=dict(color='#34c759', width=2),
            name='Health Score'
        ))
        fig_health.update_layout(
            height=150,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9)),
            yaxis=dict(showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9), range=[min(current_health_data["values"]) - 2, max(current_health_data["values"]) + 2])
        )
        st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})
        st.markdown("Overall customer satisfaction is strong, showing a positive trend this month.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Critical Alerts")
        alert_view = st.radio("View", ["Critical", "High", "Medium", "All"], horizontal=True, key="alert_view")
        st.markdown("""
        **Sudden Spike in Negative Sentiment**  
        - Mobile App Update X.Y: 45% negative  
        - Volume: 150 mentions / 3 hrs  
        - Issues: Login Failed, App Crashing  

        **High Churn Risk Pattern Detected**  
        - Pattern: Repeated Billing Errors - Savings  
        - 12 unique customer patterns  
        - Avg. sentiment: -0.8  
        """)
        st.button("View All Alerts", type="primary", key="view_alerts")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Predictive Hotspots")
        hotspot_view = st.radio("View", ["Emerging", "Trending", "Predicted"], horizontal=True, key="hotspot_view")
        st.markdown("""
        **New Overdraft Policy Confusion**  
        - Medium Impact  
        - 'Confused' Language: +30% WoW  
        - Keywords: "don't understand", "how it works"  

        **Intl. Transfer UI Issues**  
        - Low Impact  
        - Task Abandonment: +15% MoM  
        - Negative sentiment: 'Beneficiary Setup'  

        Monitor emerging confusion on overdrafts and usability for international transfers.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Prepare data for charts AND LLM context ---
    # Sentiment Data
    _temp_sentiment_values = {
        'Positive': (60 + random.random() * 10) * product_multiplier,
        'Neutral': (20 + random.random() * 5) * product_multiplier,
        'Negative': (10 + random.random() * 5) * product_multiplier
    }
    _total_sentiment = sum(_temp_sentiment_values.values()) if sum(_temp_sentiment_values.values()) > 0 else 1 # Avoid division by zero
    live_sentiment_summary_for_llm = {k: f"{(v/_total_sentiment*100):.1f}%" for k, v in _temp_sentiment_values.items()}
    sentiment_data_for_chart = pd.DataFrame({
        'Category': list(_temp_sentiment_values.keys()),
        'Value': list(_temp_sentiment_values.values())
    })

    # Intent Data
    _temp_intent_values = {
        'Info Seeking': (35 + random.random() * 10) * product_multiplier * channel_multiplier,
        'Complaint': (20 + random.random() * 5) * product_multiplier * channel_multiplier,
        'Service Request': (20 + random.random() * 5) * product_multiplier * channel_multiplier,
        'Feedback': (10 + random.random() * 5) * product_multiplier * channel_multiplier
    }
    _total_intent = sum(_temp_intent_values.values()) if sum(_temp_intent_values.values()) > 0 else 1 # Avoid division by zero
    live_intent_summary_for_llm = {k: f"{(v/_total_intent*100):.1f}% (approx {v:.0f} mentions)" for k, v in _temp_intent_values.items()}
    intent_data_for_chart = pd.DataFrame({
        'Intent': list(_temp_intent_values.keys()),
        'Value': list(_temp_intent_values.values())
    })

    # Volume Data
    _volume_data_points = [(400 + random.random() * 300 + i * 5) * channel_multiplier for i in range(30)]
    live_volume_summary_for_llm = f"Volume trend over 30 days: current day approx {int(_volume_data_points[-1])} interactions, min approx {int(min(_volume_data_points))}, max approx {int(max(_volume_data_points))}"
    vol_df_for_chart = pd.DataFrame({'Day': list(range(1, 31)), 'Volume': _volume_data_points})


    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    voice_view = st.radio("View", ["Overview", "Sentiment", "Intent", "Volume"], horizontal=True, key="voice_view")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment Distribution")
        fig_sentiment = px.pie(sentiment_data_for_chart, values='Value', names='Category', color='Category', color_discrete_map={'Positive': '#34c759', 'Neutral': '#a2a2a7', 'Negative': '#ff3b30'}, hole=0.75)
        fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=10)), showlegend=True)
        fig_sentiment.update_traces(textinfo='percent', textfont_size=10)
        st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Intent Distribution")
        fig_intent = px.bar(intent_data_for_chart, y='Intent', x='Value', orientation='h', color='Intent', color_discrete_map={'Info Seeking': '#007aff', 'Complaint': '#ff9500', 'Service Request': '#5856d6', 'Feedback': '#ffcc00'})
        fig_intent.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True), yaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True), showlegend=False)
        fig_intent.update_traces(marker_line_width=0, marker_line_color='rgba(0,0,0,0)', width=0.6)
        st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Volume Trend (30 Days)")
        fig_volume = px.line(vol_df_for_chart, x='Day', y='Volume', line_shape='spline')
        fig_volume.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines')
        fig_volume.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, tickmode='array', tickvals=[1, 5, 10, 15, 20, 25, 30], tickfont=dict(size=9)), yaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(size=9), range=[min(_volume_data_points) - 20 if _volume_data_points else 0, max(_volume_data_points) + 20 if _volume_data_points else 100]))
        st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"Positive sentiment leads at {live_sentiment_summary_for_llm.get('Positive','N/A')}. {list(live_intent_summary_for_llm.keys())[0] if live_intent_summary_for_llm else 'Info-seeking'} is a top intent. Volume shows steady increase.")


    # Top Customer Themes
    st.markdown("## Top Customer Themes")
    theme_view = st.radio("View", ["Top 10", "Trending", "Emerging", "Declining"], horizontal=True, key="theme_view")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Positive Themes")
        st.markdown("- Fast Customer Service")
        st.markdown("- Easy Mobile Banking")
        st.markdown("- Helpful Staff")
        st.markdown('> "Support resolved my issue in minutes! So efficient."')
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Negative Themes")
        st.markdown("- App Technical Issues")
        st.markdown("- Long Wait Times (Call)")
        st.markdown("- Fee Transparency")
        st.markdown('> "The app keeps crashing after the latest update. Very frustrating."')
        st.markdown('</div>', unsafe_allow_html=True)

    # Opportunity Radar
    st.markdown("## Opportunity Radar")
    opportunity_view = st.radio("View", ["High Value", "Quick Wins", "Strategic"], horizontal=True, key="opportunity_view")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **🎉 Delightful: Instant Card Activation**  
        - 75 delight mentions this week (Sentiment: +0.95)  
        - Keywords: "amazing", "so easy", "instant"  
        - Action: Amplify in marketing? Benchmark?
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **💰 Cross-Sell: Mortgage Inquiries +15%**  
        - Mortgage info seeking: +15% WoW  
        - Related: Savings, Financial Planning  
        - Action: Target with relevant mortgage info?
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""
        **⭐ Service Excellence: Complex Issues**  
        - 25 positive mentions for complex issue resolution  
        - Agents: A, B, C praised  
        - Action: Identify best practices? Recognize agents?
        """)
        st.markdown('</div>', unsafe_allow_html=True)

 # VIRA Chat Assistant
    st.markdown("## Chat with VIRA")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm VIRA, your AI assistant. How can I help with the dashboard today?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask about insights, alerts, or anything else..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            # Prepare combined dashboard state for LLM
            dashboard_state_for_llm = {
                **current_health_data, # contains score, trend, trend_label, time_period_label
                "sentiment_summary": live_sentiment_summary_for_llm,
                "intent_summary": live_intent_summary_for_llm,
                "volume_summary": live_volume_summary_for_llm,
            }

            try:
                for chunk in generate_llm_response(prompt, dashboard_state_for_llm, SYSTEM_PROMPT_VIRA):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌") # Typing effect
                message_placeholder.markdown(full_response) # Final response
            except Exception as e: # Catch any other unexpected errors from the generator
                full_response = f"An unexpected error occurred: {str(e)}"
                message_placeholder.error(full_response)

        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.markdown(f"## {page}")
    st.write("This section is under development. Please select 'Dashboard' from the sidebar to view the main dashboard.")
