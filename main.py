import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random # Kept for any potential future use, but not directly used for core data
import uuid
import json # For GCP creds
from openai import OpenAI
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# --- Google Sheets Integration ---
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
# Replace with your actual Spreadsheet ID if different
SPREADSHEET_ID = st.secrets.get("google_sheets", {}).get("spreadsheet_id", '1V5cRgnnN5GTFsD9bR05hLzsKRWkhdEy3LhuTvSnUyIM') # Example
RANGE_NAME = st.secrets.get("google_sheets", {}).get("range_name", 'sheet1!A:H') # Example

@st.cache_data(ttl=600)
def load_data_from_google_sheets():
    try:
        gcp_creds_table = st.secrets["gcp_service_account_credentials"]
        creds_info = {
            "type": gcp_creds_table["type"],
            "project_id": gcp_creds_table["project_id"],
            "private_key_id": gcp_creds_table["private_key_id"],
            "private_key": gcp_creds_table["private_key"].replace('\\n', '\n'),
            "client_email": gcp_creds_table["client_email"],
            "client_id": gcp_creds_table["client_id"],
            "auth_uri": gcp_creds_table["auth_uri"],
            "token_uri": gcp_creds_table["token_uri"],
            "auth_provider_x509_cert_url": gcp_creds_table["auth_provider_x509_cert_url"],
            "client_x509_cert_url": gcp_creds_table["client_x509_cert_url"],
            "universe_domain": gcp_creds_table["universe_domain"]
        }
        creds = service_account.Credentials.from_service_account_info(creds_info, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        values = result.get('values', [])

        if not values or len(values) < 2: # Check for header and at least one data row
            st.error("No data or only header found in the Google Sheet.")
            return pd.DataFrame(columns=['Date', 'Product', 'Channel', 'Sentimen', 'Intent', 'Interaction ID', 'Details', 'Customer ID']) # Ensure essential columns exist

        df = pd.DataFrame(values[1:], columns=values[0])

        # Standardize column names and types
        expected_columns = {
            'Date': 'datetime64[ns]',
            'Product': 'str',
            'Channel': 'str',
            'Sentimen': 'str',
            'Intent': 'str',
            'Interaction ID': 'str',
            'Details': 'str',
            'Customer ID': 'str'
        }
        # Ensure all expected columns exist, fill with NA if not
        for col, dtype in expected_columns.items():
            if col not in df.columns:
                df[col] = pd.NA
            if col == 'Date':
                 # Attempt to parse 'Date' with multiple formats
                try:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce') # General parser first
                except Exception: # If general fails, try specific
                    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')

                df.dropna(subset=['Date'], inplace=True)
            elif col in ['Product', 'Channel']:
                df[col] = df[col].astype(str).str.lower().str.replace(" ", "_").str.strip()
            elif col == 'Sentimen':
                df[col] = df[col].astype(str).str.capitalize().str.strip()
                # Standardize sentiment values
                sentiment_mapping = {
                    "Positive": "Positif", "Positive ": "Positif",
                    "Negative": "Negatif", "Negative ": "Negatif",
                    "Neutral": "Netral", "Neutral ": "Netral",
                }
                df['Sentimen'] = df['Sentimen'].replace(sentiment_mapping)
            else:
                df[col] = df[col].astype(str).str.strip()
        return df
    except KeyError as e:
        st.error(f"Missing secret: {e}. Please ensure 'gcp_service_account_credentials' and potentially 'google_sheets' config are set in your Streamlit secrets.")
        return pd.DataFrame(columns=['Date', 'Product', 'Channel', 'Sentimen', 'Intent']) # Return empty DF with expected columns
    except json.JSONDecodeError:
        st.error("Error decoding GCP credentials from Streamlit secrets. Please check the format in secrets.toml.")
        return pd.DataFrame(columns=['Date', 'Product', 'Channel', 'Sentimen', 'Intent'])
    except Exception as e:
        st.error(f"Error loading data from Google Sheets: {e}")
        # Log full error for debugging if needed: print(f"Full GSheets Error: {e}", file=sys.stderr)
        return pd.DataFrame(columns=['Date', 'Product', 'Channel', 'Sentimen', 'Intent'])

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
        height: 100%; /* Ensure cards in a row have same height */
        display: flex; /* For vertical alignment of content if needed */
        flex-direction: column; /* Stack content vertically */
    }
    .metric-card-content { /* Wrapper for content that needs to be flexible */
        flex-grow: 1;
    }
    .metric-title {
        font-size: 18px;
        font-weight: bold;
        color: #1d1d1f;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #34c759; /* Default positive, can be overridden */
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
Jika ada pertanyaan yang diluar konteks analisis anda, sampaikan bahwa itu diluar kapabilitas anda untuk menjelaskannya.

PENTING:
Sebelum memberikan jawaban akhir kepada pengguna, Anda BOLEH melakukan analisis internal atau "berpikir".
Jika Anda melakukan proses berpikir internal, *JANGAN* tuliskan pemikiran tersebut.
Jika tidak ada proses berpikir khusus atau analisis internal yang perlu dituliskan, langsung berikan jawaban.
"""
try:
    client = OpenAI(
      base_url = "https://integrate.api.nvidia.com/v1",
      api_key = st.secrets["nvidia_api"]["api_key"]
    )
except KeyError:
    st.error("NVIDIA API key not found in secrets. Please add `nvidia_api.api_key` to your secrets.toml.")
    client = None # Prevent further errors if client is not initialized
except Exception as e:
    st.error(f"Error initializing NVIDIA client: {e}")
    client = None


def generate_llm_response(user_prompt: str, dashboard_state: dict, system_prompt: str):
    if not client:
        yield "Maaf, layanan AI tidak dapat diinisialisasi. Silakan periksa konfigurasi."
        return

    dashboard_summary_for_llm = f"""
Ringkasan tampilan dasbor saat ini berdasarkan filter yang dipilih:
- Periode Waktu Terpilih: {dashboard_state.get('time_period_label_llm', 'N/A')}
- Skor Kesehatan Pelanggan (Contoh/Static): {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')} - {dashboard_state.get('trend_label', 'N/A')})

Data dari Google Sheet (berdasarkan filter saat ini):
- Total Interaksi dalam Periode: {dashboard_state.get('total_interactions', 'N/A')}
- Distribusi Sentimen: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('sentiment_summary', {}).items()]) if dashboard_state.get('sentiment_summary') and "Info" not in dashboard_state.get('sentiment_summary') else 'Tidak ada data sentimen untuk filter ini.'}.
- Distribusi Niat (Top 5): {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('intent_summary', {}).items()]) if dashboard_state.get('intent_summary') and "Info" not in dashboard_state.get('intent_summary') else 'Tidak ada data niat untuk filter ini.'}.
- Tren Volume Harian: {dashboard_state.get('volume_summary', 'N/A')}.

Informasi Dasbor Umum Lainnya (ini adalah contoh, peringatan/hotspot spesifik dapat bervariasi dan harus diperiksa pada kartunya masing-masing):
- Peringatan Kritis: (Contoh) "Lonjakan Mendadak dalam Sentimen Negatif", "Risiko Churn Tinggi".
- Hotspot Prediktif: (Contoh) "Kebingungan Kebijakan", "Masalah UI".
- Tema Pelanggan Teratas (Positif): (Contoh) "Layanan Pelanggan Cepat", "Mobile Banking Mudah".
- Tema Pelanggan Teratas (Negatif): (Contoh) "Masalah Teknis Aplikasi", "Waktu Tunggu Lama".
- Radar Peluang: (Contoh) "Fitur yang Menyenangkan", "Peluang Cross-Sell", "Keunggulan Layanan".
"""
    constructed_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{dashboard_summary_for_llm}\n\nPertanyaan Pengguna: \"{user_prompt}\""}
    ]
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1", # Or your preferred model
            messages=constructed_messages,
            temperature=0.5,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_message = f"Maaf, terjadi kesalahan saat menghubungi layanan AI: {str(e)}. Silakan coba lagi nanti atau periksa konsol."
        print(f"LLM API Error: {e}") # Log error to console for debugging
        yield error_message

# Generate health score data (remains static as per original design preference)
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
if master_df.empty and ('Date' not in master_df.columns or 'Product' not in master_df.columns or 'Channel' not in master_df.columns):
    st.warning("Master data is empty or critical columns are missing. Dashboard functionality will be limited.")

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
        available_products = sorted([p.replace("_", " ").title() for p in master_df['Product'].dropna().unique() if p])
    else:
        available_products = ["myBCA", "BCA Mobile", "KPR", "KKB"] # Fallback

    if not master_df.empty and 'Channel' in master_df.columns:
        available_channels = sorted([c.replace("_", " ").title() for c in master_df['Channel'].dropna().unique() if c])
    else:
        available_channels = ["Social Media", "Call Center", "WhatsApp"] # Fallback

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

    if not filtered_df.empty and 'Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Date']):
        today = pd.Timestamp('today').normalize()
        current_year = today.year
        current_month = today.month

        if time_period_option == "Today":
            filtered_df = filtered_df[filtered_df['Date'] == today]
        elif time_period_option == "This Week": # Monday to Sunday
            start_of_week = today - pd.to_timedelta(today.dayofweek, unit='D')
            end_of_week = start_of_week + pd.to_timedelta(6, unit='D')
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_week) & (filtered_df['Date'] <= end_of_week)]
        elif time_period_option == "This Month":
            start_of_month = today.replace(day=1)
            # end_of_month = (start_of_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1) # More robust way
            end_of_month = start_of_month + relativedelta(months=1, days=-1)
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_month) & (filtered_df['Date'] <= end_of_month)]
        elif time_period_option == "This Quarter":
            # Calculate quarter start and end
            current_quarter = (current_month - 1) // 3 + 1
            start_of_quarter = pd.Timestamp(datetime(current_year, 3 * current_quarter - 2, 1))
            end_of_quarter = start_of_quarter + relativedelta(months=3, days=-1)
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_quarter) & (filtered_df['Date'] <= end_of_quarter)]
        elif time_period_option == "This Year":
            start_of_year = today.replace(month=1, day=1)
            end_of_year = today.replace(month=12, day=31) # Correctly captures end of year
            filtered_df = filtered_df[(filtered_df['Date'] >= start_of_year) & (filtered_df['Date'] <= end_of_year)]
        # "All Periods" means no date filtering beyond initial load
    elif time_period_option != "All Periods" and ('Date' not in filtered_df.columns or not pd.api.types.is_datetime64_any_dtype(filtered_df['Date'])):
        st.caption(f"Warning: 'Date' column missing or not in datetime format for '{time_period_option}' filtering. Showing all available data for other filters.")
    elif master_df.empty:
         st.caption("No data loaded, filtering cannot be applied.")


    if "All Products" not in selected_products_display and selected_products_display:
        selected_products_internal = [p.lower().replace(" ", "_") for p in selected_products_display]
        if 'Product' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Product'].isin(selected_products_internal)]
        elif not master_df.empty: # Only show warning if master_df wasn't empty to begin with
            st.caption("Warning: 'Product' column not available for filtering.")

    if "All Channels" not in selected_channels_display and selected_channels_display:
        selected_channels_internal = [c.lower().replace(" ", "_") for c in selected_channels_display]
        if 'Channel' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Channel'].isin(selected_channels_internal)]
        elif not master_df.empty:
            st.caption("Warning: 'Channel' column not available for filtering.")

    # Health score data (static, selected by time_period_option)
    health_score_data_source = generate_health_score_data()
    time_period_map_health = {
        "All Periods": "all", "Today": "today", "This Week": "week",
        "This Month": "month", "This Quarter": "quarter", "This Year": "year"
    }
    selected_time_key_health = time_period_map_health.get(time_period_option, "month")
    current_health_data = health_score_data_source.get(selected_time_key_health, health_score_data_source["month"]).copy()
    current_health_data['time_period_label_display'] = time_period_option # For display on the card

    # --- Dashboard widgets ---
    st.markdown("## Dashboard Widgets")
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Customer Health Score</span>", unsafe_allow_html=True)
        # health_view = st.radio("View", ["Real-time", "Daily Trend", "Comparison"], horizontal=True, key="health_view") # Removed for simplicity, can be added back

        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f'<div class="metric-value" style="color: #34c759;">{current_health_data["score"]}%</div>', unsafe_allow_html=True)
        with score_col2:
            trend_icon = "↑" if current_health_data["trend_positive"] else "↓"
            trend_class = "metric-trend-positive" if current_health_data["trend_positive"] else "metric-trend-negative"
            st.markdown(f'<div class="{trend_class}">{trend_icon} {current_health_data["trend"]} {current_health_data["trend_label"]}</div>', unsafe_allow_html=True)

        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True) # For chart
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
        st.markdown("</div>", unsafe_allow_html=True) # Close metric-card-content
        st.markdown("<small>Overall customer satisfaction index. Trend shown vs. previous period.</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


    with row1_col2: # Critical Alerts (static content from design)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Critical Alerts</span>", unsafe_allow_html=True)
        alert_view = st.radio("View", ["Critical", "High", "Medium", "All"], horizontal=True, key="alert_view_r1c2") # Unique key
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("""
        <small>
        **Sudden Spike in Negative Sentiment**<br>
        - Mobile App Update X.Y: 45% negative<br>
        - Volume: 150 mentions / 3 hrs<br>
        - Issues: Login Failed, App Crashing<br><br>
        **High Churn Risk Pattern Detected**<br>
        - Pattern: Repeated Billing Errors - Savings<br>
        - 12 unique customer patterns<br>
        - Avg. sentiment: -0.8
        </small>
        """)
        st.markdown("</div>", unsafe_allow_html=True) # Close metric-card-content
        st.button("View All Alerts", type="primary", key="view_alerts_r1c2") # Unique key
        st.markdown('</div>', unsafe_allow_html=True)

    with row1_col3: # Predictive Hotspots (static content from design)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Predictive Hotspots</span>", unsafe_allow_html=True)
        hotspot_view = st.radio("View", ["Emerging", "Trending", "Predicted"], horizontal=True, key="hotspot_view_r1c3") # Unique key
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("""
        <small>
        **New Overdraft Policy Confusion**<br>
        - Medium Impact<br>
        - 'Confused' Language: +30% WoW<br>
        - Keywords: "don't understand", "how it works"<br><br>
        **Intl. Transfer UI Issues**<br>
        - Low Impact<br>
        - Task Abandonment: +15% MoM<br>
        - Negative sentiment: 'Beneficiary Setup'
        </small>
        """)
        st.markdown("</div>", unsafe_allow_html=True) # Close metric-card-content
        st.markdown("<small>Monitor emerging confusion and usability issues.</small>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Prepare data for charts AND LLM context from filtered_df ---
    total_interactions_for_llm = len(filtered_df) if not filtered_df.empty else 0
    no_data_df = pd.DataFrame({'Label': ['No Data'], 'Value': [1]}) # For empty charts

    # Sentiment Data
    if not filtered_df.empty and 'Sentimen' in filtered_df.columns and not filtered_df['Sentimen'].dropna().empty:
        sentiment_counts = filtered_df['Sentimen'].value_counts()
        sentiment_data_for_chart = sentiment_counts.reset_index()
        sentiment_data_for_chart.columns = ['Category', 'Value']
        _total_sentiment = sentiment_counts.sum()
        live_sentiment_summary_for_llm = {k: f"{(v/_total_sentiment*100):.1f}% ({v} mentions)" for k, v in sentiment_counts.items()} if _total_sentiment > 0 else {}
    else:
        sentiment_data_for_chart = pd.DataFrame({'Category': ['No Data'], 'Value': [1]})
        live_sentiment_summary_for_llm = {"Info": "No sentiment data for current filter."}

    # Intent Data
    if not filtered_df.empty and 'Intent' in filtered_df.columns and not filtered_df['Intent'].dropna().empty:
        intent_counts = filtered_df['Intent'].value_counts().nlargest(5) # Top 5 intents
        intent_data_for_chart = intent_counts.reset_index()
        intent_data_for_chart.columns = ['Intent', 'Value']
        _total_intent_top5 = intent_counts.sum()
        live_intent_summary_for_llm = {k: f"{(v/_total_intent_top5*100):.1f}% ({v} mentions)" for k, v in intent_counts.items()} if _total_intent_top5 > 0 else {}
    else:
        intent_data_for_chart = pd.DataFrame({'Intent': ['No Data'], 'Value': [1]})
        live_intent_summary_for_llm = {"Info": "No intent data for current filter."}

    # Volume Data
    _volume_data_points = [0, 1] # Default for y-axis range if no data
    if not filtered_df.empty and 'Date' in filtered_df.columns and pd.api.types.is_datetime64_any_dtype(filtered_df['Date']) and not filtered_df['Date'].dropna().empty:
        volume_over_time = filtered_df.groupby(filtered_df['Date'].dt.date)['Date'].count()
        vol_df_for_chart = volume_over_time.reset_index(name='Volume')
        vol_df_for_chart.columns = ['Day', 'Volume']
        vol_df_for_chart['Day'] = pd.to_datetime(vol_df_for_chart['Day']) # Ensure 'Day' is datetime for Plotly
        if not vol_df_for_chart.empty:
            _volume_data_points = vol_df_for_chart['Volume'].tolist()
            live_volume_summary_for_llm = f"Min daily {vol_df_for_chart['Volume'].min()}, Max daily {vol_df_for_chart['Volume'].max()}, Avg daily {vol_df_for_chart['Volume'].mean():.1f}. Total {vol_df_for_chart['Volume'].sum()} interactions in period."
        else: # Filtered resulted in empty data for volume trend
            vol_df_for_chart = pd.DataFrame({'Day': [pd.Timestamp('today').normalize()], 'Volume': [0]})
            live_volume_summary_for_llm = "No volume data to display for the selected date range after filtering."
    else:
        vol_df_for_chart = pd.DataFrame({'Day': [pd.Timestamp('today').normalize()], 'Volume': [0]}) # Placeholder for empty chart
        live_volume_summary_for_llm = "Volume data cannot be trended (Date column missing, not datetime, or no data after filtering)."


    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    # voice_view = st.radio("View", ["Overview", "Sentiment", "Intent", "Volume"], horizontal=True, key="voice_view_snapshot") # Removed for simplicity
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Sentiment Distribution</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        if not sentiment_data_for_chart.empty and sentiment_data_for_chart['Category'].iloc[0] != 'No Data':
            fig_sentiment = px.pie(sentiment_data_for_chart, values='Value', names='Category', color='Category',
                                   color_discrete_map={'Positif': '#34c759', 'Netral': '#a2a2a7', 'Negatif': '#ff3b30', 'Unknown': '#cccccc'},
                                   hole=0.7)
            fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=5, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.25, xanchor='center', x=0.5, font=dict(size=10)), showlegend=True)
            fig_sentiment.update_traces(textinfo='percent', textfont_size=10, insidetextorientation='radial') # textinfo='percent+label' can be too crowded
        else:
            fig_sentiment = go.Figure(go.Indicator(mode="number", value=0, number={'font':{'size':1}}, title={"text": "No Sentiment Data", "font":{"size":12}})) # Smaller font
            fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Intent Distribution (Top 5)</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        if not intent_data_for_chart.empty and intent_data_for_chart['Intent'].iloc[0] != 'No Data':
            intent_color_map = {'Informasi': '#007aff', 'Keluhan': '#ff9500', 'Permohonan': '#5856d6', 'Layanan umum': '#ffcc00', 'Penutupan': '#ff3b30', 'Saran': '#34c759', 'Apresiasi': '#af52de'}
            fig_intent = px.bar(intent_data_for_chart, y='Intent', x='Value', orientation='h', color='Intent',
                                color_discrete_map=intent_color_map,
                                color_discrete_sequence=px.colors.qualitative.Pastel) # Fallback colors
            fig_intent.update_layout(height=230, margin=dict(l=0, r=10, t=5, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(size=9)), yaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, tickfont=dict(size=9), categoryorder='total ascending'), showlegend=False)
            fig_intent.update_traces(marker_line_width=0, marker_line_color='rgba(0,0,0,0)', width=0.6, texttemplate='%{x}', textposition='outside')
        else:
            fig_intent = go.Figure(go.Indicator(mode="number", value=0, number={'font':{'size':1}}, title={"text": "No Intent Data", "font":{"size":12}}))
            fig_intent.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with row2_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"<span class='metric-title'>Volume Trend ({time_period_option})</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        if not vol_df_for_chart.empty and vol_df_for_chart['Volume'].sum() > 0 :
            fig_volume = px.line(vol_df_for_chart, x='Day', y='Volume', line_shape='spline', markers=False) # Markers can be noisy for many points
            fig_volume.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines')
            y_min_vol = 0 if not _volume_data_points or min(_volume_data_points) < 0 else min(_volume_data_points) # Ensure y_min is not negative
            y_max_vol = 10 if not _volume_data_points else max(_volume_data_points)
            # Add a small buffer to y-axis, ensure min is 0 if all values are 0 or positive.
            y_range_min = 0 if y_min_vol == 0 and y_max_vol == 0 else max(0, y_min_vol - (y_max_vol - y_min_vol) * 0.1 -1)
            y_range_max = y_max_vol + (y_max_vol - y_min_vol) * 0.1 + 1
            if y_range_min == y_range_max: y_range_max +=1 # Ensure range is not zero


            fig_volume.update_layout(height=230, margin=dict(l=0, r=10, t=5, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    xaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, tickfont=dict(size=9)),
                                    yaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(size=9), range=[y_range_min, y_range_max]))
        else:
            fig_volume = go.Figure(go.Indicator(mode="number", value=0, number={'font':{'size':1}}, title={"text": "No Volume Data", "font":{"size":12}}))
            fig_volume.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary text below charts
    summary_text_parts = []
    if "Info" not in live_sentiment_summary_for_llm and live_sentiment_summary_for_llm:
        top_sentiment_cat = max(live_sentiment_summary_for_llm, key=lambda k: float(live_sentiment_summary_for_llm[k].split('%')[0]))
        summary_text_parts.append(f"{top_sentiment_cat} sentiment leads at {live_sentiment_summary_for_llm[top_sentiment_cat].split(' ')[0]}.")
    if "Info" not in live_intent_summary_for_llm and live_intent_summary_for_llm:
        top_intent = list(live_intent_summary_for_llm.keys())[0] # Already sorted by nlargest
        summary_text_parts.append(f"'{top_intent}' is a top intent.")
    if "Total" in live_volume_summary_for_llm: # Check if volume summary is meaningful
         summary_text_parts.append(f"Volume shows {total_interactions_for_llm} total interactions for the period.")

    if summary_text_parts:
        st.markdown(" ".join(summary_text_parts))
    elif total_interactions_for_llm == 0:
        st.markdown("No interaction data found for the current filter selection.")
    else: # Some interactions but no sentiment/intent perhaps
        st.markdown(f"Displaying data for {total_interactions_for_llm} interactions based on current filters. Some metrics might be unavailable.")


    # Top Customer Themes (Static content from design)
    st.markdown("## Top Customer Themes")
    # theme_view = st.radio("View", ["Top 10", "Trending", "Emerging", "Declining"], horizontal=True, key="theme_view_snapshot") # Removed for simplicity
    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Top Positive Themes</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("<small>- Fast Customer Service</small>", unsafe_allow_html=True)
        st.markdown("<small>- Easy Mobile Banking</small>", unsafe_allow_html=True)
        st.markdown("<small>- Helpful Staff</small>", unsafe_allow_html=True)
        st.markdown('<small>> <i>"Support resolved my issue in minutes! So efficient."</i></small>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row3_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>Top Negative Themes</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("<small>- App Technical Issues</small>", unsafe_allow_html=True)
        st.markdown("<small>- Long Wait Times (Call)</small>", unsafe_allow_html=True)
        st.markdown("<small>- Fee Transparency</small>", unsafe_allow_html=True)
        st.markdown('<small>> <i>"The app keeps crashing after the latest update. Very frustrating."</i></small>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Opportunity Radar (Static content from design)
    st.markdown("## Opportunity Radar")
    # opportunity_view = st.radio("View", ["High Value", "Quick Wins", "Strategic"], horizontal=True, key="opportunity_view_snapshot") # Removed for simplicity
    row4_col1, row4_col2, row4_col3 = st.columns(3)
    with row4_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>🎉 Delightful</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("""<small>
        **Instant Card Activation**<br>
        - 75 delight mentions this week (Sentiment: +0.95)<br>
        - Keywords: "amazing", "so easy", "instant"<br>
        - Action: Amplify in marketing? Benchmark?
        </small>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row4_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>💰 Cross-Sell</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("""<small>
        **Mortgage Inquiries +15%**<br>
        - Mortgage info seeking: +15% WoW<br>
        - Related: Savings, Financial Planning<br>
        - Action: Target with relevant mortgage info?
        </small>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with row4_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("<span class='metric-title'>⭐ Service Excellence</span>", unsafe_allow_html=True)
        st.markdown('<div class="metric-card-content">', unsafe_allow_html=True)
        st.markdown("""<small>
        **Complex Issues Resolved**<br>
        - 25 positive mentions for complex issue resolution<br>
        - Agents: A, B, C praised<br>
        - Action: Identify best practices? Recognize agents?
        </small>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
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

            # Prepare combined dashboard state for LLM
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
            except Exception as e: # Catch any other unexpected errors from the generator
                full_response = f"Terjadi kesalahan tak terduga saat menghasilkan respons: {str(e)}"
                message_placeholder.error(full_response)
                print(f"Chat Error: {e}") # Log to console

        st.session_state.messages.append({"role": "assistant", "content": full_response})

else:
    st.markdown(f"## {page}")
    st.write("Bagian ini sedang dalam pengembangan. Silakan pilih 'Dashboard' dari sidebar untuk melihat dasbor utama.")
