import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
import uuid
from openai import OpenAI # Ensure openai library is installed: pip install openai
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta # For more complex date manipulations

# --- Google Sheets Integration ---
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Constants for Google Sheets
# SERVICE_ACCOUNT_FILE = 'key.json' # No longer needed
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = '1V5cRgnnN5GTFsD9bR05hLzsKRWkhdEy3LhuTvSnUyIM' # Your spreadsheet ID
RANGE_NAME = 'sheet1!A:H'

@st.cache_data(ttl=600)
def load_data_from_google_sheets():
    """Loads data from Google Sheets using credentials from Streamlit Secrets."""
    try:
        # Load the entire JSON string from secrets and parse it
        creds_json_str = st.secrets["gcp_service_account_credentials"]
        creds_info = json.loads(creds_json_str) # Parse the string into a Python dictionary

        creds = service_account.Credentials.from_service_account_info(
            creds_info, scopes=SCOPES) # Use from_service_account_info

        service = build('sheets', 'v4', credentials=creds)

        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            st.error("No data found in the Google Sheet.")
            return pd.DataFrame()
        else:
            df = pd.DataFrame(values[1:], columns=values[0])
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
                df.dropna(subset=['Date'], inplace=True)
            else:
                st.warning("Column 'Date' not found in Google Sheet. Time filtering will not work correctly.")
            if 'Product' in df.columns:
                df['Product'] = df['Product'].astype(str).str.lower().str.replace(" ", "_")
            if 'Channel' in df.columns:
                df['Channel'] = df['Channel'].astype(str).str.lower().str.replace(" ", "_")
            if 'Sentimen' in df.columns:
                df['Sentimen'] = df['Sentimen'].astype(str).str.capitalize()
            if 'Intent' in df.columns:
                df['Intent'] = df['Intent'].astype(str)
            return df
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please ensure 'gcp_service_account_credentials' is set in your Streamlit secrets.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        st.error("Error decoding GCP credentials from Streamlit secrets. Please check the format in secrets.toml.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        return pd.DataFrame()

# Set page configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling (same as before)
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
SYSTEM_PROMPT_VIRA = """
Anda adalah VIRA, seorang konsultan virtual untuk Bank BCA.
Tugas utama Anda adalah menganalisis data dasbor yang disediakan dan memberikan wawasan, ringkasan, serta saran yang relevan.
Fokuslah pada metrik seperti skor kesehatan (jika ada), tren, sentimen pelanggan, niat panggilan, dan volume panggilan berdasarkan data yang disaring.
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
    dashboard_summary_for_llm = f"""
Ringkasan tampilan dasbor saat ini berdasarkan filter yang dipilih:
- Periode Waktu Terpilih: {dashboard_state.get('time_period_label_llm', 'N/A')}
- Skor Kesehatan Pelanggan (Contoh): {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')} - {dashboard_state.get('trend_label', 'N/A')})

Data dari Google Sheet (berdasarkan filter saat ini):
- Total Interaksi dalam Periode: {dashboard_state.get('total_interactions', 'N/A')}
- Distribusi Sentimen: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('sentiment_summary', {}).items()]) if dashboard_state.get('sentiment_summary') else 'Tidak ada data sentimen untuk filter ini.'}.
- Distribusi Niat: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('intent_summary', {}).items()]) if dashboard_state.get('intent_summary') else 'Tidak ada data niat untuk filter ini.'}.
- Tren Volume: {dashboard_state.get('volume_summary', 'N/A')}.

Informasi Dasbor Umum Lainnya (ini adalah contoh, peringatan/hotspot spesifik dapat bervariasi dan harus diperiksa pada kartunya masing-masing):
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
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=constructed_messages,
            temperature=0.5, # Adjusted for more factual responses
            top_p=0.7,       # Adjusted
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_message = f"Maaf, terjadi kesalahan saat menghubungi layanan AI: {str(e)}. Silakan coba lagi nanti atau periksa konsol."
        print(f"LLM API Error: {e}")
        yield error_message

# Generate health score data (remains static for now)
def generate_health_score_data():
    return {
        "today": {"labels": ["9 AM", "11 AM", "1 PM", "3 PM", "5 PM", "7 PM", "9 PM"], "values": [78, 76, 80, 79, 81, 83, 84], "score": 84, "trend": "+2.5%", "trend_positive": True, "trend_label": "vs. yesterday"},
        "week": {"labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "values": [79, 78, 80, 81, 83, 84, 85], "score": 85, "trend": "+1.8%", "trend_positive": True, "trend_label": "vs. last week"},
        "month": {"labels": ["Week 1", "Week 2", "Week 3", "Week 4"], "values": [79, 80, 81, 82], "score": 82, "trend": "+1.5%", "trend_positive": True, "trend_label": "vs. last month"},
        "quarter": {"labels": ["Jan", "Feb", "Mar"], "values": [76, 79, 83], "score": 83, "trend": "+3.2%", "trend_positive": True, "trend_label": "vs. last quarter"},
        "year": {"labels": ["Q1", "Q2", "Q3", "Q4"], "values": [75, 77, 80, 84], "score": 84, "trend": "+4.1%", "trend_positive": True, "trend_label": "vs. last year"},
        "all": {"labels": ["2019", "2020", "2021", "2022", "2023", "2024"], "values": [73, 71, 75, 78, 80, 83], "score": 83, "trend": "+10.4%", "trend_positive": True, "trend_label": "over 5 years"},
    }

# --- Load Data ---
master_df = load_data_from_google_sheets()

# Sidebar (same as before)
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

    # --- FILTERS ---
    col1, col2, col3 = st.columns(3)
    with col1:
        time_period_option = st.selectbox(
            "TIME",
            ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
            index=3, # Default to "This Month"
            key="time_filter"
        )

    # Get unique products and channels from the master_df for filter options
    if not master_df.empty and 'Product' in master_df.columns:
        available_products = sorted(list(master_df['Product'].str.replace("_", " ").str.title().unique()))
    else:
        available_products = ["myBCA", "BCA Mobile", "KPR", "KKB", "KSM", "Investasi", "Asuransi", "KMK", "Kartu Kredit", "EDC & QRIS", "Poket Valas"] # Fallback

    if not master_df.empty and 'Channel' in master_df.columns:
        available_channels = sorted(list(master_df['Channel'].str.replace("_", " ").str.title().unique()))
    else:
        available_channels = ["Social Media", "Call Center", "WhatsApp", "Webchat", "VIRA", "E-mail", "Survey Gallup", "Survey BSQ", "Survey CX"] # Fallback

    with col2:
        selected_products_display = st.multiselect(
            "PRODUCT",
            ["All Products"] + available_products,
            default=["All Products"],
            key="product_filter"
        )
    with col3:
        selected_channels_display = st.multiselect(
            "CHANNEL",
            ["All Channels"] + available_channels,
            default=["All Channels"],
            key="channel_filter"
        )

    # --- FILTERING LOGIC ---
    filtered_df = master_df.copy()

    if not filtered_df.empty and 'Date' in filtered_df.columns:
        today = pd.Timestamp('today').normalize()
        if time_period_option == "Today":
            filtered_df = filtered_df[filtered_df['Date'] == today]
        elif time_period_option == "This Week":
            start_of_week = today - pd.to_timedelta(today.dayofweek, unit='D')
            end_of_week = start_of_week + pd.to_timedelta(6, unit='D')
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_week) & (filtered_df['Date'] <= end_of_week)]
        elif time_period_option == "This Month":
            start_of_month = today.replace(day=1)
            end_of_month = (start_of_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_month) & (filtered_df['Date'] <= end_of_month)]
        elif time_period_option == "This Quarter":
            start_of_quarter = today.to_period('Q').start_time
            end_of_quarter = today.to_period('Q').end_time
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_quarter) & (filtered_df['Date'] <= end_of_quarter)]
        elif time_period_option == "This Year":
            start_of_year = today.replace(month=1, day=1)
            end_of_year = today.replace(month=12, day=31)
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_year) & (filtered_df['Date'] <= end_of_year)]
        # "All Periods" means no date filtering beyond initial load
    elif time_period_option != "All Periods":
        st.caption(f"Warning: 'Date' column not available for '{time_period_option}' filtering. Showing all data.")


    if "All Products" not in selected_products_display and selected_products_display:
        selected_products_internal = [p.lower().replace(" ", "_") for p in selected_products_display]
        if 'Product' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Product'].isin(selected_products_internal)]
        else:
            st.caption("Warning: 'Product' column not available for filtering.")


    if "All Channels" not in selected_channels_display and selected_channels_display:
        selected_channels_internal = [c.lower().replace(" ", "_") for c in selected_channels_display]
        if 'Channel' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Channel'].isin(selected_channels_internal)]
        else:
            st.caption("Warning: 'Channel' column not available for filtering.")

    # Health score data (static for now)
    health_score_data_source = generate_health_score_data()
    time_period_map_health = {
        "All Periods": "all", "Today": "today", "This Week": "week",
        "This Month": "month", "This Quarter": "quarter", "This Year": "year"
    }
    selected_time_key_health = time_period_map_health.get(time_period_option, "month")
    current_health_data = health_score_data_source.get(selected_time_key_health, health_score_data_source["month"]).copy()
    current_health_data['time_period_label'] = time_period_option

    # --- Dashboard widgets ---
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
            x=current_health_data["labels"], y=current_health_data["values"], mode='lines', fill='tozeroy',
            fillcolor='rgba(52,199,89,0.18)', line=dict(color='#34c759', width=2), name='Health Score'
        ))
        fig_health.update_layout(
            height=150, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9)),
            yaxis=dict(showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9), range=[min(current_health_data["values"]) - 2, max(current_health_data["values"]) + 2])
        )
        st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})
        st.markdown("Overall customer satisfaction is strong, showing a positive trend this month.") # Example text
        st.markdown('</div>', unsafe_allow_html=True)

    with col2: # Critical Alerts (static content)
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

    with col3: # Predictive Hotspots (static content)
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

    # --- Prepare data for charts AND LLM context from filtered_df ---
    total_interactions_for_llm = len(filtered_df)

    # Sentiment Data
    if not filtered_df.empty and 'Sentimen' in filtered_df.columns:
        sentiment_counts = filtered_df['Sentimen'].value_counts()
        sentiment_data_for_chart = sentiment_counts.reset_index()
        sentiment_data_for_chart.columns = ['Category', 'Value']
        _total_sentiment = sentiment_counts.sum()
        live_sentiment_summary_for_llm = {k: f"{(v/_total_sentiment*100):.1f}% ({v} mentions)" for k, v in sentiment_counts.items()} if _total_sentiment > 0 else {}
    else:
        sentiment_data_for_chart = pd.DataFrame({'Category': ['No Data'], 'Value': [1]})
        live_sentiment_summary_for_llm = {"Info": "No sentiment data for current filter."}

    # Intent Data
    if not filtered_df.empty and 'Intent' in filtered_df.columns:
        intent_counts = filtered_df['Intent'].value_counts().nlargest(5) # Top 5 intents
        intent_data_for_chart = intent_counts.reset_index()
        intent_data_for_chart.columns = ['Intent', 'Value']
        _total_intent = intent_counts.sum() # Using sum of top 5 for this summary
        live_intent_summary_for_llm = {k: f"{(v/_total_intent*100):.1f}% ({v} mentions)" for k, v in intent_counts.items()} if _total_intent > 0 else {}
    else:
        intent_data_for_chart = pd.DataFrame({'Intent': ['No Data'], 'Value': [1]})
        live_intent_summary_for_llm = {"Info": "No intent data for current filter."}

    # Volume Data
    if not filtered_df.empty and 'Date' in filtered_df.columns:
        # Group by day for volume trend, regardless of selected period (could be refined)
        volume_over_time = filtered_df.groupby(filtered_df['Date'].dt.date)['Date'].count()
        vol_df_for_chart = volume_over_time.reset_index(name='Volume')
        vol_df_for_chart.columns = ['Day', 'Volume']
        if not vol_df_for_chart.empty:
            live_volume_summary_for_llm = f"Volume trend over selected period: Min daily {vol_df_for_chart['Volume'].min()}, Max daily {vol_df_for_chart['Volume'].max()}, Avg daily {vol_df_for_chart['Volume'].mean():.1f}. Total {vol_df_for_chart['Volume'].sum()} interactions."
            _volume_data_points = vol_df_for_chart['Volume'].tolist() # for y-axis range
        else:
            live_volume_summary_for_llm = "No volume data to display for the selected filters."
            _volume_data_points = [0, 1] # Default for range
            vol_df_for_chart = pd.DataFrame({'Day': [pd.Timestamp('today').date()], 'Volume': [0]}) # Placeholder for empty chart
    else:
        vol_df_for_chart = pd.DataFrame({'Day': [pd.Timestamp('today').date()], 'Volume': [0]}) # Placeholder
        live_volume_summary_for_llm = "Volume data cannot be trended (Date column missing or no data)."
        _volume_data_points = [0, 1]


    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    voice_view = st.radio("View", ["Overview", "Sentiment", "Intent", "Volume"], horizontal=True, key="voice_view")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment Distribution")
        if not sentiment_data_for_chart.empty and sentiment_data_for_chart['Category'].iloc[0] != 'No Data':
            fig_sentiment = px.pie(sentiment_data_for_chart, values='Value', names='Category', color='Category',
                                   color_discrete_map={'Positif': '#34c759', 'Netral': '#a2a2a7', 'Negatif': '#ff3b30', 'Unknown': '#cccccc'},
                                   hole=0.75)
            fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=10)), showlegend=True)
            fig_sentiment.update_traces(textinfo='percent+label', textfont_size=10, insidetextorientation='radial')
        else:
            fig_sentiment = go.Figure(go.Indicator(mode="number", value=0, title={"text": "No Sentiment Data"}))
            fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Intent Distribution (Top 5)")
        if not intent_data_for_chart.empty and intent_data_for_chart['Intent'].iloc[0] != 'No Data':
            # Define a broader color map or use a continuous scale if many intents
            intent_color_map = {'Informasi': '#007aff', 'Keluhan': '#ff9500', 'Permohonan': '#5856d6', 'Layanan umum': '#ffcc00', 'Penutupan': '#ff3b30'}
            fig_intent = px.bar(intent_data_for_chart, y='Intent', x='Value', orientation='h', color='Intent',
                                color_discrete_map=intent_color_map) # Add more colors if needed or use px.colors.qualitative.Plotly
            fig_intent.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True), yaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, categoryorder='total ascending'), showlegend=False)
            fig_intent.update_traces(marker_line_width=0, marker_line_color='rgba(0,0,0,0)', width=0.6)
        else:
            fig_intent = go.Figure(go.Indicator(mode="number", value=0, title={"text": "No Intent Data"}))
            fig_intent.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"### Volume Trend ({time_period_option})")
        if not vol_df_for_chart.empty and vol_df_for_chart['Volume'].sum() > 0 :
            fig_volume = px.line(vol_df_for_chart, x='Day', y='Volume', line_shape='spline', markers=True)
            fig_volume.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines+markers')
            y_min_vol = 0 if not _volume_data_points else min(_volume_data_points)
            y_max_vol = 10 if not _volume_data_points else max(_volume_data_points)
            y_range_vol = [max(0, y_min_vol - (y_max_vol-y_min_vol)*0.1), y_max_vol + (y_max_vol-y_min_vol)*0.1 + 1] # Ensure some padding

            fig_volume.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, tickfont=dict(size=9)),
                                    yaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(size=9), range=y_range_vol))
        else:
            fig_volume = go.Figure(go.Indicator(mode="number", value=0, title={"text": "No Volume Data"}))
            fig_volume.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary text below charts
    if not filtered_df.empty:
        summary_parts = []
        if 'Positif' in live_sentiment_summary_for_llm:
             summary_parts.append(f"Sentimen Positif dominan sebesar {live_sentiment_summary_for_llm['Positif'].split(' ')[0]}.")
        if live_intent_summary_for_llm and "Info" not in live_intent_summary_for_llm:
             top_intent = list(live_intent_summary_for_llm.keys())[0]
             summary_parts.append(f"Niat '{top_intent}' menjadi yang teratas.")
        if "Total" in live_volume_summary_for_llm:
            summary_parts.append(f"{live_volume_summary_for_llm.split('.')[1].strip()}.") # Get the total part
        st.markdown(" ".join(summary_parts) if summary_parts else "Tidak ada data yang cukup untuk ringkasan.")
    else:
        st.markdown("Tidak ada data untuk filter yang dipilih.")


    # Top Customer Themes (Static content, can be made dynamic later)
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

    # Opportunity Radar (Static content, can be made dynamic later)
    st.markdown("## Opportunity Radar")
    opportunity_view = st.radio("View", ["High Value", "Quick Wins", "Strategic"], horizontal=True, key="opportunity_view")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""**🎉 Delightful: Instant Card Activation**<br>- 75 delight mentions this week (Sentiment: +0.95)<br>- Keywords: "amazing", "so easy", "instant"<br>- Action: Amplify in marketing? Benchmark?""")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""**💰 Cross-Sell: Mortgage Inquiries +15%**<br>- Mortgage info seeking: +15% WoW<br>- Related: Savings, Financial Planning<br>- Action: Target with relevant mortgage info?""")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("""**⭐ Service Excellence: Complex Issues**<br>- 25 positive mentions for complex issue resolution<br>- Agents: A, B, C praised<br>- Action: Identify best practices? Recognize agents?""")
        st.markdown('</div>', unsafe_allow_html=True)

    # VIRA Chat Assistant
    st.markdown("## Chat with VIRA")
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Saya VIRA, asisten AI Anda. Ada yang bisa saya bantu terkait data di dasbor hari ini?"}
        ]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Tanyakan tentang wawasan, peringatan, atau hal lainnya..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            dashboard_state_for_llm = {
                **current_health_data, # contains score, trend, trend_label (from static health score)
                "time_period_label_llm": time_period_option, # Actual filter selection for LLM
                "total_interactions": total_interactions_for_llm,
                "sentiment_summary": live_sentiment_summary_for_llm,
                "intent_summary": live_intent_summary_for_llm,
                "volume_summary": live_volume_summary_for_llm,
            }

            try:
                stream = generate_llm_response(prompt, dashboard_state_for_llm, SYSTEM_PROMPT_VIRA)
                for chunk in stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌") # Typing effect
                message_placeholder.markdown(full_response) # Final response
            except Exception as e:
                full_response = f"Terjadi kesalahan tak terduga saat menghasilkan respons: {str(e)}"
                message_placeholder.error(full_response)
                print(f"Chat Error: {e}")


        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.markdown(f"## {page}")
    st.write("Bagian ini sedang dalam pengembangan. Silakan pilih 'Dashboard' dari sidebar untuk melihat dasbor utama.")
