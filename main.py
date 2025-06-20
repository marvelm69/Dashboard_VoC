import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
# import random # No longer needed for core data
# import uuid # No longer needed for core data
from openai import OpenAI

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.auth.exceptions import RefreshError

# ------------------ GOOGLE SHEETS API SETUP -----------------------
SERVICE_ACCOUNT_FILE = 'key.json' # Make sure this file exists or update path
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly'] # Readonly is safer if you only read
SPREADSHEET_ID = '1V5cRgnnN5GTFsD9bR05hLzsKRWkhdEy3LhuTvSnUyIM' # Your Spreadsheet ID

@st.cache_data(ttl=600) # Cache data for 10 minutes
def get_google_sheets_service():
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        return service
    except FileNotFoundError:
        st.error(f"Service account key file '{SERVICE_ACCOUNT_FILE}' not found. Please ensure it's in the correct path.")
        return None
    except RefreshError as e:
        st.error(f"Error with Google Sheets credentials (RefreshError): {e}. Ensure the service account is correctly set up and has access to the sheet.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Sheets authentication: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_sheet_data(_service, sheet_name, range_name): # Tambahkan '_' di depan service
    if _service is None: # Gunakan _service di dalam fungsi
        return []
    try:
        result = _service.spreadsheets().values().get( # Gunakan _service di dalam fungsi
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!{range_name}"
        ).execute()
        values = result.get('values', [])
        return values
    except Exception as e:
        st.warning(f"Could not fetch data from sheet '{sheet_name}', range '{range_name}'. Error: {e}")
        return []

# --- Data Processing Functions from Google Sheets ---
@st.cache_data(ttl=600)
def process_health_score_data_from_sheet(raw_data):
    health_scores = {}
    if not raw_data or len(raw_data) < 2: # Header + at least one data row
        st.warning("HealthScores sheet data is empty or malformed.")
        return generate_fallback_health_score_data() # Fallback to some default

    header = raw_data[0]
    try:
        # Expected header: TimePeriodKey, Labels, Values, Score, Trend, TrendPositive, TrendLabel
        key_idx = header.index('TimePeriodKey')
        labels_idx = header.index('Labels (comma-separated)')
        values_idx = header.index('Values (comma-separated)')
        score_idx = header.index('Score')
        trend_idx = header.index('Trend')
        trend_pos_idx = header.index('TrendPositive (TRUE/FALSE)')
        trend_label_idx = header.index('TrendLabel')
    except ValueError:
        st.error("HealthScores sheet has incorrect headers. Expected: TimePeriodKey, Labels (comma-separated), Values (comma-separated), Score, Trend, TrendPositive (TRUE/FALSE), TrendLabel")
        return generate_fallback_health_score_data()

    for row in raw_data[1:]:
        if len(row) < max(key_idx, labels_idx, values_idx, score_idx, trend_idx, trend_pos_idx, trend_label_idx) + 1:
            continue # Skip malformed rows

        time_period_key = row[key_idx]
        try:
            labels = [label.strip() for label in row[labels_idx].split(',')]
            values = [int(val.strip()) for val in row[values_idx].split(',')]
            score = int(row[score_idx])
            trend_positive = row[trend_pos_idx].strip().upper() == 'TRUE'

            health_scores[time_period_key] = {
                "labels": labels,
                "values": values,
                "score": score,
                "trend": row[trend_idx],
                "trend_positive": trend_positive,
                "trend_label": row[trend_label_idx],
            }
        except ValueError as e:
            st.warning(f"Skipping row in HealthScores due to data conversion error for '{time_period_key}': {e}. Row: {row}")
            continue
        except IndexError:
            st.warning(f"Skipping row in HealthScores due to missing columns for '{time_period_key}'. Row: {row}")
            continue

    if not health_scores: # If all rows failed or sheet was empty after header
        return generate_fallback_health_score_data()
    return health_scores

@st.cache_data(ttl=600)
def process_categorical_data_from_sheet(raw_data, category_col_name="Category", value_col_name="Value"):
    if not raw_data or len(raw_data) < 2:
        return pd.DataFrame({category_col_name: [], value_col_name: []}) # Empty DataFrame with correct columns

    header = raw_data[0]
    try:
        cat_idx = header.index(category_col_name)
        val_idx = header.index(value_col_name)
    except ValueError:
        st.error(f"Sheet for categorical data has incorrect headers. Expected: '{category_col_name}', '{value_col_name}'")
        return pd.DataFrame({category_col_name: [], value_col_name: []})

    data_dict = {category_col_name: [], value_col_name: []}
    for row in raw_data[1:]:
        if len(row) > max(cat_idx, val_idx):
            try:
                data_dict[category_col_name].append(row[cat_idx])
                data_dict[value_col_name].append(float(row[val_idx])) # Ensure value is float
            except (ValueError, IndexError) as e:
                st.warning(f"Skipping row in categorical data due to error: {e}. Row: {row}")
                continue
    return pd.DataFrame(data_dict)

@st.cache_data(ttl=600)
def process_volume_data_from_sheet(raw_data):
    if not raw_data or len(raw_data) < 2:
        return pd.DataFrame({'Day': [], 'Volume': []})

    header = raw_data[0]
    try:
        day_idx = header.index('Day')
        vol_idx = header.index('Volume')
    except ValueError:
        st.error("VolumeData sheet has incorrect headers. Expected: 'Day', 'Volume'")
        return pd.DataFrame({'Day': [], 'Volume': []})

    data_dict = {'Day': [], 'Volume': []}
    for row in raw_data[1:]:
        if len(row) > max(day_idx, vol_idx):
            try:
                data_dict['Day'].append(int(row[day_idx]))
                data_dict['Volume'].append(float(row[vol_idx]))
            except (ValueError, IndexError) as e:
                st.warning(f"Skipping row in VolumeData due to error: {e}. Row: {row}")
                continue
    return pd.DataFrame(data_dict)

@st.cache_data(ttl=600)
def process_textual_list_from_sheet(raw_data, num_cols=1):
    """ Processes sheets where each row is an item, potentially with multiple details. """
    if not raw_data or len(raw_data) < 2: # Header + data
        return []

    items = []
    for row in raw_data[1:]: # Skip header
        if row: # If row is not empty
            # Take up to num_cols, fill with empty strings if fewer
            items.append([row[i] if i < len(row) else "" for i in range(num_cols)])
    return items

# Fallback data if sheet fetching fails
def generate_fallback_health_score_data():
    st.warning("Using fallback data for Health Scores.")
    return {
        "month": { # Default to month if specific key not found later
            "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
            "values": [70, 72, 71, 73],
            "score": 73,
            "trend": "+0.5%",
            "trend_positive": True,
            "trend_label": "vs. last month (fallback)",
        }
    }
def generate_fallback_categorical_data(category_name="Category", value_name="Value"):
    st.warning(f"Using fallback data for {category_name}.")
    return pd.DataFrame({category_name: ["Default A", "Default B"], value_name: [60, 40]})

def generate_fallback_volume_data():
    st.warning("Using fallback data for Volume Trend.")
    return pd.DataFrame({'Day': list(range(1, 11)), 'Volume': [100 + i*5 for i in range(10)]})

def generate_fallback_textual_list(item_name="Item", num_details=1):
    st.warning(f"Using fallback data for {item_name}.")
    if num_details == 1:
        return [["Default " + item_name + " 1"], ["Default " + item_name + " 2"]]
    else:
        return [["Default " + item_name + " Title 1"] + [f"Detail {i+1}" for i in range(num_details-1)],
                ["Default " + item_name + " Title 2"] + [f"Detail {i+1}" for i in range(num_details-1)]]


# Initialize Google Sheets Service
sheets_service = get_google_sheets_service()

# Set page configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (same as your original)
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
        min-height: 250px; /* Ensure cards have a minimum height */
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

# IMPORTANT: Use Streamlit secrets for API keys in production!
# For local development, you can set it as an environment variable
# or temporarily hardcode it (NOT RECOMMENDED FOR PRODUCTION).
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "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") # Replace with your actual key if not using secrets

if not NVIDIA_API_KEY or NVIDIA_API_KEY == "YOUR_NVIDIA_API_KEY_HERE":
    st.error("NVIDIA API Key not configured. Please set it in Streamlit secrets (key: NVIDIA_API_KEY) or environment variables.")
    client = None
else:
    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = NVIDIA_API_KEY
    )

def generate_llm_response(user_prompt: str, dashboard_state: dict, system_prompt: str):
    if client is None:
        yield "Layanan AI tidak dikonfigurasi. Silakan periksa API Key."
        return

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
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=constructed_messages,
            temperature=1.00,
            top_p=0.01,
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


# Sidebar (same as your original)
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

    # Filters (same as your original, currently they don't re-fetch data, just for display/LLM context)
    col1, col2, col3 = st.columns(3)
    with col1:
        time_period_display = st.selectbox(
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

    # --- FETCH DATA FROM GOOGLE SHEETS ---
    if sheets_service:
        raw_health_data = fetch_sheet_data(sheets_service, "HealthScores", "A:G")
        health_score_data_source = process_health_score_data_from_sheet(raw_health_data)

        raw_sentiment_data = fetch_sheet_data(sheets_service, "SentimentData", "A:B")
        sentiment_data_for_chart = process_categorical_data_from_sheet(raw_sentiment_data, "Category", "Value")

        raw_intent_data = fetch_sheet_data(sheets_service, "IntentData", "A:B")
        intent_data_for_chart = process_categorical_data_from_sheet(raw_intent_data, "Intent", "Value")

        raw_volume_data = fetch_sheet_data(sheets_service, "VolumeData", "A:B")
        vol_df_for_chart = process_volume_data_from_sheet(raw_volume_data)

        raw_alerts_data = fetch_sheet_data(sheets_service, "CriticalAlertsData", "A:D") # Assuming up to 4 columns
        critical_alerts_list = process_textual_list_from_sheet(raw_alerts_data, num_cols=4)

        raw_hotspots_data = fetch_sheet_data(sheets_service, "PredictiveHotspotsData", "A:D") # Assuming up to 4 columns
        predictive_hotspots_list = process_textual_list_from_sheet(raw_hotspots_data, num_cols=4)

        raw_positive_themes = fetch_sheet_data(sheets_service, "PositiveThemesData", "A:B") # Theme, Quote
        positive_themes_list = process_textual_list_from_sheet(raw_positive_themes, num_cols=2)

        raw_negative_themes = fetch_sheet_data(sheets_service, "NegativeThemesData", "A:B") # Theme, Quote
        negative_themes_list = process_textual_list_from_sheet(raw_negative_themes, num_cols=2)

        raw_opportunities = fetch_sheet_data(sheets_service, "OpportunityRadarData", "A:E") # Category, Title, D1, D2, Action
        opportunity_radar_list = process_textual_list_from_sheet(raw_opportunities, num_cols=5)

    else: # Fallback if sheets_service is None
        st.error("Google Sheets service could not be initialized. Displaying fallback data.")
        health_score_data_source = generate_fallback_health_score_data()
        sentiment_data_for_chart = generate_fallback_categorical_data("Category", "Value")
        intent_data_for_chart = generate_fallback_categorical_data("Intent", "Value")
        vol_df_for_chart = generate_fallback_volume_data()
        critical_alerts_list = generate_fallback_textual_list("Critical Alert", num_details=3)
        predictive_hotspots_list = generate_fallback_textual_list("Predictive Hotspot", num_details=3)
        positive_themes_list = generate_fallback_textual_list("Positive Theme", num_details=2)
        negative_themes_list = generate_fallback_textual_list("Negative Theme", num_details=2)
        opportunity_radar_list = generate_fallback_textual_list("Opportunity", num_details=5)


    # Filter logic for health score
    time_period_map = {
        "All Periods": "all", "Today": "today", "This Week": "week",
        "This Month": "month", "This Quarter": "quarter", "This Year": "year"
    }
    selected_time_key = time_period_map.get(time_period_display, "month")

    # Get current health data, fall back to 'month' or first available if selected key not present
    current_health_data = health_score_data_source.get(selected_time_key)
    if not current_health_data:
        current_health_data = health_score_data_source.get("month", health_score_data_source.get(next(iter(health_score_data_source)))) # Fallback chain
        st.warning(f"Data for '{time_period_display}' not found in HealthScores sheet. Displaying fallback or 'month' data.")

    current_health_data['time_period_label'] = time_period_display


    # --- Dashboard widgets ---
    st.markdown("## Dashboard Widgets")
    col1_dash, col2_dash, col3_dash = st.columns(3)

    with col1_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Customer Health Score")
        health_view = st.radio("View", ["Real-time", "Daily Trend", "Comparison"], horizontal=True, key="health_view")

        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f'<div class="metric-value">{current_health_data.get("score", "N/A")}%</div>', unsafe_allow_html=True)
        with score_col2:
            trend_positive = current_health_data.get("trend_positive", False)
            trend_icon = "↑" if trend_positive else "↓"
            trend_class = "metric-trend-positive" if trend_positive else "metric-trend-negative"
            st.markdown(f'<div class="{trend_class}">{trend_icon} {current_health_data.get("trend", "N/A")} {current_health_data.get("trend_label", "")}</div>', unsafe_allow_html=True)

        fig_health = go.Figure()
        fig_health.add_trace(go.Scatter(
            x=current_health_data.get("labels", []),
            y=current_health_data.get("values", []),
            mode='lines', fill='tozeroy', fillcolor='rgba(52,199,89,0.18)',
            line=dict(color='#34c759', width=2), name='Health Score'
        ))
        min_val = min(current_health_data.get("values", [0])) if current_health_data.get("values") else 0
        max_val = max(current_health_data.get("values", [100])) if current_health_data.get("values") else 100
        fig_health.update_layout(
            height=150, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9)),
            yaxis=dict(showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9), range=[min_val - 2, max_val + 2])
        )
        st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})
        st.markdown("Overall customer satisfaction based on selected period.") # Generic description
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Critical Alerts")
        alert_view = st.radio("View", ["Critical", "High", "Medium", "All"], horizontal=True, key="alert_view") # View selection not implemented for brevity
        if critical_alerts_list:
            for alert in critical_alerts_list:
                st.markdown(f"**{alert[0]}**") # Title
                for detail in alert[1:]:
                    if detail: st.markdown(f"- {detail}")
                st.markdown("---")
        else:
            st.info("No critical alerts data found or loaded.")
        st.button("View All Alerts", type="primary", key="view_alerts")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Predictive Hotspots")
        hotspot_view = st.radio("View", ["Emerging", "Trending", "Predicted"], horizontal=True, key="hotspot_view") # View selection not implemented
        if predictive_hotspots_list:
            for hotspot in predictive_hotspots_list:
                st.markdown(f"**{hotspot[0]}**") # Title
                for detail in hotspot[1:]:
                    if detail: st.markdown(f"- {detail}")
                st.markdown("---")
        else:
            st.info("No predictive hotspots data found or loaded.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Prepare data for charts AND LLM context (using fetched data) ---
    if not sentiment_data_for_chart.empty:
        _total_sentiment = sentiment_data_for_chart['Value'].sum() if sentiment_data_for_chart['Value'].sum() > 0 else 1
        live_sentiment_summary_for_llm = {
            row['Category']: f"{(row['Value']/_total_sentiment*100):.1f}%"
            for index, row in sentiment_data_for_chart.iterrows()
        }
    else:
        live_sentiment_summary_for_llm = {"Positive": "N/A", "Neutral": "N/A", "Negative": "N/A"}

    if not intent_data_for_chart.empty:
        _total_intent = intent_data_for_chart['Value'].sum() if intent_data_for_chart['Value'].sum() > 0 else 1
        live_intent_summary_for_llm = {
            row['Intent']: f"{(row['Value']/_total_intent*100):.1f}% (approx {row['Value']:.0f} mentions)"
            for index, row in intent_data_for_chart.iterrows()
        }
    else:
        live_intent_summary_for_llm = {"Info Seeking": "N/A"}

    if not vol_df_for_chart.empty:
        _volume_data_points = vol_df_for_chart['Volume'].tolist()
        live_volume_summary_for_llm = f"Volume trend over {len(_volume_data_points)} days: current day approx {int(_volume_data_points[-1]) if _volume_data_points else 'N/A'} interactions, min approx {int(min(_volume_data_points)) if _volume_data_points else 'N/A'}, max approx {int(max(_volume_data_points)) if _volume_data_points else 'N/A'}"
    else:
        _volume_data_points = []
        live_volume_summary_for_llm = "Volume data N/A"


    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    voice_view = st.radio("View", ["Overview", "Sentiment", "Intent", "Volume"], horizontal=True, key="voice_view")
    col1_snap, col2_snap, col3_snap = st.columns(3)

    with col1_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment Distribution")
        if not sentiment_data_for_chart.empty:
            fig_sentiment = px.pie(sentiment_data_for_chart, values='Value', names='Category', color='Category',
                                   color_discrete_map={'Positive': '#34c759', 'Neutral': '#a2a2a7', 'Negative': '#ff3b30'}, hole=0.75)
            fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=10)), showlegend=True)
            fig_sentiment.update_traces(textinfo='percent', textfont_size=10)
            st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Sentiment data not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Intent Distribution")
        if not intent_data_for_chart.empty:
            fig_intent = px.bar(intent_data_for_chart, y='Intent', x='Value', orientation='h', color='Intent',
                                color_discrete_map={'Info Seeking': '#007aff', 'Complaint': '#ff9500', 'Service Request': '#5856d6', 'Feedback': '#ffcc00'})
            fig_intent.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True),
                                     yaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True), showlegend=False)
            fig_intent.update_traces(marker_line_width=0, marker_line_color='rgba(0,0,0,0)', width=0.6)
            st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Intent data not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"### Volume Trend ({len(_volume_data_points)} Days)")
        if not vol_df_for_chart.empty:
            fig_volume = px.line(vol_df_for_chart, x='Day', y='Volume', line_shape='spline')
            fig_volume.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines')
            min_vol = min(_volume_data_points) - 20 if _volume_data_points else 0
            max_vol = max(_volume_data_points) + 20 if _volume_data_points else 100
            fig_volume.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)',
                                     xaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, tickmode='auto', tickfont=dict(size=9)),
                                     yaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(size=9), range=[min_vol, max_vol]))
            st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("Volume data not available.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"Positive sentiment leads at {live_sentiment_summary_for_llm.get('Positive','N/A')}. {list(live_intent_summary_for_llm.keys())[0] if live_intent_summary_for_llm else 'Info-seeking'} is a top intent. Volume shows trends from sheet data.")


    # Top Customer Themes
    st.markdown("## Top Customer Themes")
    theme_view = st.radio("View", ["Top 10", "Trending", "Emerging", "Declining"], horizontal=True, key="theme_view")
    col1_theme, col2_theme = st.columns(2)

    with col1_theme:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Positive Themes")
        if positive_themes_list:
            for theme_item in positive_themes_list:
                st.markdown(f"- {theme_item[0]}") # Theme
                if len(theme_item) > 1 and theme_item[1]: # Optional Quote
                    st.markdown(f'> "{theme_item[1]}"')
        else:
            st.info("No positive themes data found.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_theme:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Negative Themes")
        if negative_themes_list:
            for theme_item in negative_themes_list:
                st.markdown(f"- {theme_item[0]}") # Theme
                if len(theme_item) > 1 and theme_item[1]: # Optional Quote
                    st.markdown(f'> "{theme_item[1]}"')
        else:
            st.info("No negative themes data found.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Opportunity Radar
    st.markdown("## Opportunity Radar")
    opportunity_view = st.radio("View", ["High Value", "Quick Wins", "Strategic"], horizontal=True, key="opportunity_view")

    if opportunity_radar_list:
        num_opportunities = len(opportunity_radar_list)
        cols_opportunity = st.columns(min(num_opportunities, 3)) # Max 3 columns, or fewer if less data

        for i, opportunity in enumerate(opportunity_radar_list):
            if i < 3 : # Display up to 3
                with cols_opportunity[i % 3]:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    # opportunity: [Category, Title, Detail1, Detail2, Action]
                    category_icon = {"Delightful": "🎉", "Cross-Sell": "💰", "Service Excel": "⭐"}.get(opportunity[0], "💡")
                    st.markdown(f"**{category_icon} {opportunity[0]}: {opportunity[1]}**") # Category: Title
                    if opportunity[2]: st.markdown(f"- {opportunity[2]}") # Detail1
                    if opportunity[3]: st.markdown(f"- {opportunity[3]}") # Detail2
                    if opportunity[4]: st.markdown(f"- Action: {opportunity[4]}") # Action
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No opportunity radar data found.")


    # VIRA Chat Assistant
    st.markdown("## Chat with VIRA")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm VIRA, your AI assistant. How can I help with the dashboard today?"}]

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
            dashboard_state_for_llm = {
                **current_health_data,
                "sentiment_summary": live_sentiment_summary_for_llm,
                "intent_summary": live_intent_summary_for_llm,
                "volume_summary": live_volume_summary_for_llm,
            }
            try:
                for chunk in generate_llm_response(prompt, dashboard_state_for_llm, SYSTEM_PROMPT_VIRA):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            except Exception as e:
                full_response = f"An unexpected error occurred with LLM: {str(e)}"
                message_placeholder.error(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

else: # Other pages
    st.markdown(f"## {page}")
    st.write("This section is under development. Please select 'Dashboard' from the sidebar to view the main dashboard.")
