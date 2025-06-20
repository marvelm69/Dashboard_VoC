import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from collections import Counter

from googleapiclient.discovery import build
from google.oauth2 import service_account
from google.auth.exceptions import RefreshError
from datetime import datetime, timedelta

# ------------------ GOOGLE SHEETS API SETUP -----------------------
SERVICE_ACCOUNT_FILE = 'key.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = '1V5cRgnnN5GTFsD9bR05hLzsKRWkhdEy3LhuTvSnUyIM' # Sesuaikan jika perlu

@st.cache_data(ttl=600)
def get_google_sheets_service():
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        return service
    except FileNotFoundError:
        st.error(f"Service account key file '{SERVICE_ACCOUNT_FILE}' not found.")
        return None
    except RefreshError as e:
        st.error(f"Error with Google Sheets credentials: {e}.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Sheets authentication: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_main_data(_service, sheet_name="MainData", range_name="A:H"): # Ambil kolom A sampai H
    if _service is None:
        return pd.DataFrame() # Kembalikan DataFrame kosong jika service tidak ada
    try:
        result = _service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!{range_name}"
        ).execute()
        values = result.get('values', [])
        if not values or len(values) < 1: # Jika tidak ada data atau hanya header kosong
            st.warning(f"No data found in sheet '{sheet_name}'.")
            return pd.DataFrame()

        header = values[0]
        data_rows = values[1:]

        # Pastikan semua baris data memiliki jumlah kolom yang sama dengan header
        # Jika tidak, isi dengan None atau string kosong untuk menghindari error saat membuat DataFrame
        processed_rows = []
        num_cols = len(header)
        for row in data_rows:
            if len(row) < num_cols:
                processed_rows.append(row + [None] * (num_cols - len(row)))
            elif len(row) > num_cols:
                processed_rows.append(row[:num_cols]) # Potong jika lebih
            else:
                processed_rows.append(row)

        df = pd.DataFrame(processed_rows, columns=header)

        # Konversi Tipe Data Penting
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            # df = df.dropna(subset=['Date']) # Hapus baris dengan tanggal yang tidak valid
        if 'Sentimen' in df.columns:
            df['Sentimen'] = df['Sentimen'].astype(str).str.capitalize() # Pastikan konsisten (Positif, Negatif, Netral)
        if 'Intent' in df.columns:
            df['Intent'] = df['Intent'].astype(str)
        if 'Product' in df.columns:
            df['Product'] = df['Product'].astype(str)
        if 'Channel' in df.columns:
            df['Channel'] = df['Channel'].astype(str)

        return df
    except Exception as e:
        st.error(f"Could not fetch or process data from sheet '{sheet_name}'. Error: {e}")
        return pd.DataFrame()


# --- Data Processing Functions from Main DataFrame ---

def filter_dataframe(df, time_period_display, selected_products, selected_channels):
    if df.empty:
        return df

    # Filter Tanggal
    # Konversi time_period_display ke rentang tanggal yang sesuai
    # Ini adalah contoh sederhana, Anda mungkin perlu logika yang lebih kompleks
    # untuk 'This Week', 'This Quarter', dll. yang bergantung pada tanggal saat ini.
    # Untuk 'All Periods', tidak ada filter tanggal.
    end_date = df['Date'].max() # atau datetime.now() jika ingin relatif terhadap hari ini

    if time_period_display == "Today":
        start_date = end_date.replace(hour=0, minute=0, second=0, microsecond=0)
        df_filtered = df[df['Date'] >= start_date]
    elif time_period_display == "This Week": # Asumsi minggu dimulai Senin
        start_date = end_date - timedelta(days=end_date.weekday())
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif time_period_display == "This Month":
        start_date = end_date.replace(day=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif time_period_display == "This Quarter":
        current_quarter = (end_date.month - 1) // 3 + 1
        first_month_of_quarter = 3 * (current_quarter - 1) + 1
        start_date = end_date.replace(month=first_month_of_quarter, day=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif time_period_display == "This Year":
        start_date = end_date.replace(month=1, day=1)
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    elif time_period_display == "All Periods":
        df_filtered = df.copy()
    else: # Default to all if no specific period matches
        df_filtered = df.copy()

    # Filter Produk
    if "All Products" not in selected_products and selected_products:
        df_filtered = df_filtered[df_filtered['Product'].isin(selected_products)]

    # Filter Channel
    if "All Channels" not in selected_channels and selected_channels:
        df_filtered = df_filtered[df_filtered['Channel'].isin(selected_channels)]

    return df_filtered


def get_sentiment_distribution(df):
    if df.empty or 'Sentimen' not in df.columns:
        return pd.DataFrame({'Category': ['Positive', 'Neutral', 'Negative'], 'Value': [0, 0, 0]})
    sentiment_counts = df['Sentimen'].value_counts().reindex(['Positif', 'Netral', 'Negatif'], fill_value=0)
    return pd.DataFrame({'Category': sentiment_counts.index, 'Value': sentiment_counts.values})

def get_intent_distribution(df):
    if df.empty or 'Intent' not in df.columns:
        return pd.DataFrame({'Intent': [], 'Value': []})
    intent_counts = df['Intent'].value_counts()
    return pd.DataFrame({'Intent': intent_counts.index, 'Value': intent_counts.values})

def get_volume_trend(df, days=30):
    if df.empty or 'Date' not in df.columns:
        return pd.DataFrame({'Day': [], 'Volume': []})

    # Ambil data untuk 'days' terakhir dari tanggal maksimum di data
    max_date_in_data = df['Date'].max()
    start_date = max_date_in_data - timedelta(days=days-1)

    volume_data = df[df['Date'] >= start_date].groupby(df['Date'].dt.date)['Date'].count().reset_index(name='Volume')
    volume_data.rename(columns={'Date': 'Day_Date'}, inplace=True) # Ganti nama agar tidak bentrok

    # Buat rentang tanggal penuh untuk 'days' terakhir untuk memastikan tidak ada hari yang hilang
    all_days = pd.date_range(start=start_date, end=max_date_in_data, freq='D').to_series().dt.date
    all_days_df = pd.DataFrame(all_days, columns=['Day_Date'])

    volume_data_full = pd.merge(all_days_df, volume_data, on='Day_Date', how='left').fillna(0)
    volume_data_full['Day'] = range(1, len(volume_data_full) + 1) # Label hari 1 sampai 'days'

    return volume_data_full[['Day', 'Volume']]


def get_top_themes(df, sentiment_filter, top_n=3):
    if df.empty or 'Summary' not in df.columns or 'Sentimen' not in df.columns:
        return [] # Format: [["Theme/Summary 1", "Quote (optional)"], ...]

    filtered_df = df[df['Sentimen'] == sentiment_filter]
    if filtered_df.empty:
        return []

    # Menggunakan 'Summary' sebagai tema. Jika ada kolom 'Keywords' atau 'Topics' akan lebih baik.
    # Untuk contoh ini, kita ambil summary yang paling sering muncul.
    # Anda mungkin ingin logika yang lebih canggih di sini (misalnya, NLP untuk ekstraksi tema).
    theme_counts = filtered_df['Summary'].value_counts().nlargest(top_n)

    themes_list = []
    for summary, count in theme_counts.items():
        # Mencari salah satu 'masked_text' sebagai contoh quote
        example_masked_text = filtered_df[filtered_df['Summary'] == summary]['masked_text'].iloc[0] if not filtered_df[filtered_df['Summary'] == summary].empty else ""
        themes_list.append([summary, f'"..." (masked text example)']) # Menandakan ini dari masked_text
    return themes_list

def calculate_health_score_data(df, time_period_key="month"):
    """
    Ini adalah fungsi yang paling perlu disesuaikan.
    Bagaimana skor kesehatan dihitung dari data mentah Anda?
    Contoh: Rata-rata sentimen numerik (misal Positif=1, Netral=0, Negatif=-1)
    dikonversi ke persentase.
    """
    if df.empty:
        return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "Data unavailable"
        }

    # --- CONTOH SEDERHANA Perhitungan Skor Kesehatan ---
    # Ubah sentimen menjadi nilai numerik
    sentiment_map = {'Positif': 1, 'Netral': 0.5, 'Negatif': 0} # Skala 0-1
    df['SentimentScore'] = df['Sentimen'].map(sentiment_map).fillna(0.5) # Default netral jika tidak termapping

    # Untuk tren harian/mingguan, kita perlu data historis skor harian
    # Mari kita hitung skor rata-rata harian untuk periode tertentu
    if 'Date' not in df.columns:
         return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "Date column missing"
        }

    daily_scores = df.groupby(df['Date'].dt.date)['SentimentScore'].mean().reset_index()
    daily_scores['SentimentScore'] = daily_scores['SentimentScore'] * 100 # Jadi persentase

    # Logika untuk labels dan values berdasarkan time_period_key
    # Ini akan sangat bergantung pada bagaimana Anda ingin menampilkan tren
    # Contoh untuk 'month': skor rata-rata mingguan dalam sebulan terakhir

    if daily_scores.empty:
         return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "No daily scores"
        }

    current_score = daily_scores['SentimentScore'].iloc[-1] if not daily_scores.empty else 0

    # Untuk tren, bandingkan dengan skor sebelumnya (misal, hari sebelumnya atau periode sebelumnya)
    previous_score = daily_scores['SentimentScore'].iloc[-2] if len(daily_scores) > 1 else current_score
    trend_value = current_score - previous_score
    trend_percentage_str = f"{trend_value:+.1f}%" if previous_score != 0 else "+0.0%" # Hindari bagi dengan nol

    # Ini contoh labels dan values untuk tampilan "This Month" dengan data harian
    # Anda perlu menyesuaikan ini untuk "Today", "This Week", dll.
    # Atau, Anda bisa selalu menampilkan, misalnya, 30 titik data terakhir

    num_points_to_show = 30
    trend_labels = [d.strftime('%b %d') for d in daily_scores['Date'].tail(num_points_to_show)]
    trend_values = daily_scores['SentimentScore'].tail(num_points_to_show).tolist()

    if not trend_labels: # Fallback jika tidak ada data
        trend_labels = ["N/A"]
        trend_values = [0]
        current_score = 0

    return {
        "labels": trend_labels,
        "values": trend_values,
        "score": int(round(current_score)),
        "trend": trend_percentage_str,
        "trend_positive": trend_value >= 0,
        "trend_label": "vs. previous period", # Sesuaikan
    }


# --- (Fungsi untuk Critical Alerts, Predictive Hotspots, Opportunity Radar perlu definisi baru) ---
# Untuk sekarang, kita buat mereka mengembalikan list kosong atau data placeholder dari sheet jika ada
# Atau Anda bisa membuat sheet terpisah untuk ini dan menggunakan fetch_sheet_data seperti sebelumnya.

@st.cache_data(ttl=600)
def fetch_supplementary_data(_service, sheet_name, range_name, num_cols=1):
    if _service is None: return []
    try:
        result = _service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{sheet_name}!{range_name}").execute()
        values = result.get('values', [])
        if not values or len(values) < 2: return [] # Header + data

        items = []
        for row in values[1:]: # Skip header
            if row: items.append([row[i] if i < len(row) else "" for i in range(num_cols)])
        return items
    except Exception as e:
        st.warning(f"Could not fetch data from supplementary sheet '{sheet_name}'. Error: {e}")
        return []

# Initialize Google Sheets Service
sheets_service = get_google_sheets_service()

# Set page configuration
st.set_page_config(page_title="Voice of Customer Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS (sama)
st.markdown("""...""") # CSS Anda di sini (tidak saya sertakan ulang untuk keringkasan)

# --- NVIDIA API Client Initialization ---
SYSTEM_PROMPT_VIRA = """...""" # Prompt Anda di sini
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY", "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")
client = None
# ... (Inisialisasi client OpenAI seperti sebelumnya) ...
if not NVIDIA_API_KEY or NVIDIA_API_KEY == "YOUR_NVIDIA_API_KEY_HERE": # Ganti dengan placeholder jika perlu
    st.error("NVIDIA API Key not configured.")
else:
    try:
        client = OpenAI(
          base_url = "https://integrate.api.nvidia.com/v1",
          api_key = NVIDIA_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing OpenAI (NVIDIA) client: {e}")

def generate_llm_response(user_prompt: str, dashboard_state: dict, system_prompt: str):
    # ... (Fungsi LLM Anda seperti sebelumnya) ...
    if client is None:
        yield "Layanan AI tidak dikonfigurasi."
        return
    # (sisanya sama)
    dashboard_summary_for_llm = f"""
Ringkasan tampilan dasbor saat ini berdasarkan filter yang dipilih:
- Periode Waktu Terpilih untuk Skor Kesehatan: {dashboard_state.get('time_period_label', 'N/A')}
- Skor Kesehatan Pelanggan: {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')} - {dashboard_state.get('trend_label', 'N/A')})

Ringkasan Grafik Langsung (perkiraan berdasarkan filter saat ini):
- Distribusi Sentimen: Positif: {dashboard_state.get('sentiment_summary', {}).get('Positif', 'N/A')}, Netral: {dashboard_state.get('sentiment_summary', {}).get('Neutral', 'N/A')}, Negatif: {dashboard_state.get('sentiment_summary', {}).get('Negatif', 'N/A')}.
- Distribusi Niat: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('intent_summary', {}).items()]) if dashboard_state.get('intent_summary') else 'N/A'}.
- Tren Volume: {dashboard_state.get('volume_summary', 'N/A')}.
""" # (Sesuaikan summary ini juga jika perlu)
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
        error_message = f"Maaf, terjadi kesalahan saat menghubungi layanan AI: {str(e)}."
        print(f"LLM API Error: {e}")
        yield error_message

# Sidebar
with st.sidebar:
    st.title("VOCAL")
    st.markdown("---")
    st.header("Menu")
    if "menu_nav" not in st.session_state:
        st.session_state.menu_nav = "Dashboard"
    page = st.selectbox("Navigate", ["Dashboard", "Analytics", "Feedback", "Alerts", "Reports"], key="menu_nav")
    # ... (Sisa sidebar sama) ...
    st.header("Customer Insights")
    st.selectbox("Insights", ["Sentiment Analysis", "Journey Mapping", "Satisfaction Scores", "Theme Analysis"], key="insights_nav")
    st.header("Operations")
    st.selectbox("Operations", ["Real-time Monitoring", "Predictive Analytics", "Performance Metrics", "Action Items"], key="ops_nav")
    st.header("Configuration")
    st.selectbox("Config", ["Settings", "User Management", "Security", "Help & Support"], key="config_nav")
    st.markdown("---")
    st.markdown("**Sebastian**")
    st.markdown("CX Manager")


# Fetch Main Data Once
main_df_raw = pd.DataFrame()
if sheets_service:
    main_df_raw = fetch_main_data(sheets_service, sheet_name="sheet1") # Ganti "MainData" jika perlu
else:
    st.error("Google Sheets service not available. Cannot load main data.")
    # Anda bisa load data contoh dari file CSV di sini sebagai fallback jika diperlukan
    # main_df_raw = pd.read_csv("fallback_main_data.csv") 

if main_df_raw.empty and sheets_service: # Jika fetch gagal tapi service ada
    st.warning("Failed to load data from Google Sheets. Dashboard might be empty or show errors.")


# Main content
if page == "Dashboard":
    st.title("Customer Experience Health")
    st.markdown("Real-time Insights & Performance Overview")

    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        # Ambil daftar produk dan channel unik dari data jika ada
        unique_products = ["All Products"] + sorted(main_df_raw['Product'].astype(str).unique().tolist()) if not main_df_raw.empty and 'Product' in main_df_raw.columns else ["All Products"]
        unique_channels = ["All Channels"] + sorted(main_df_raw['Channel'].astype(str).unique().tolist()) if not main_df_raw.empty and 'Channel' in main_df_raw.columns else ["All Channels"]

        time_period_display = st.selectbox(
            "TIME",
            ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
            index=3, key="time_filter"
        )
    with col2:
        products_filter = st.multiselect("PRODUCT", unique_products, default=["All Products"], key="product_filter")
    with col3:
        channels_filter = st.multiselect("CHANNEL", unique_channels, default=["All Channels"], key="channel_filter")

    # Filter Data Utama berdasarkan pilihan filter
    df_filtered = main_df_raw.copy() # Mulai dengan data mentah (atau yang sudah difilter jika Anda implementasi filter_dataframe)
    if not main_df_raw.empty:
        df_filtered = filter_dataframe(main_df_raw, time_period_display, products_filter, channels_filter)
    else:
        st.info("No data loaded to apply filters.")


    # --- AGREGASI DATA DARI df_filtered ---
    sentiment_data_for_chart = get_sentiment_distribution(df_filtered)
    intent_data_for_chart = get_intent_distribution(df_filtered)
    vol_df_for_chart = get_volume_trend(df_filtered, days=30) # Ambil 30 hari tren

    # Untuk Health Score, kita perlu data yang sesuai dengan time_period_display
    # Fungsi calculate_health_score_data perlu disesuaikan untuk ini
    # Atau, kita bisa selalu menghitung berdasarkan df_filtered secara keseluruhan untuk 'score' saat ini
    # dan trennya dihitung secara internal oleh fungsi tersebut.
    current_health_data = calculate_health_score_data(df_filtered, time_period_key=time_period_display.lower().replace("this ","")) # e.g. "month"
    current_health_data['time_period_label'] = time_period_display

    # Top Themes
    positive_themes_list = get_top_themes(df_filtered, "Positif", top_n=3)
    negative_themes_list = get_top_themes(df_filtered, "Negatif", top_n=3)

    # Critical Alerts, Predictive Hotspots, Opportunity Radar
    # Ini masih menggunakan supplementary sheets. Anda bisa ganti logika ini
    # untuk menganalisis df_filtered atau membuat rules engine sederhana.
    critical_alerts_list = []
    predictive_hotspots_list = []
    opportunity_radar_list = []
    if sheets_service:
        critical_alerts_list = fetch_supplementary_data(sheets_service, "CriticalAlertsData", "A:D", num_cols=4)
        predictive_hotspots_list = fetch_supplementary_data(sheets_service, "PredictiveHotspotsData", "A:D", num_cols=4)
        opportunity_radar_list = fetch_supplementary_data(sheets_service, "OpportunityRadarData", "A:E", num_cols=5)
    else: # Fallback jika sheet tambahan tidak bisa di-load
        critical_alerts_list = [["Fallback Alert: High Negative Sentiment Spike", "Details unavailable"]]
        predictive_hotspots_list = [["Fallback Hotspot: Policy Confusion", "Details unavailable"]]
        opportunity_radar_list = [["Fallback Opp: Feature Request", "Category: Delightful", "Details..."]]


    # --- Dashboard widgets ---
    st.markdown("## Dashboard Widgets")
    col1_dash, col2_dash, col3_dash = st.columns(3)

    with col1_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Customer Health Score")
        # health_view = st.radio("View", ["Real-time", "Daily Trend", "Comparison"], horizontal=True, key="health_view") # Opsi view ini perlu implementasi lebih lanjut

        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f'<div class="metric-value">{current_health_data.get("score", "N/A")}%</div>', unsafe_allow_html=True)
        with score_col2:
            trend_positive = current_health_data.get("trend_positive", False)
            trend_icon = "↑" if trend_positive else "↓"
            trend_class = "metric-trend-positive" if trend_positive else "metric-trend-negative"
            st.markdown(f'<div class="{trend_class}">{trend_icon} {current_health_data.get("trend", "N/A")} {current_health_data.get("trend_label", "")}</div>', unsafe_allow_html=True)

        fig_health = go.Figure()
        if current_health_data.get("values"):
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
        st.markdown("Customer health based on sentiment analysis of interactions.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Critical Alerts")
        # alert_view = st.radio("View", ["Critical", "High", "Medium", "All"], horizontal=True, key="alert_view")
        if critical_alerts_list:
            for alert in critical_alerts_list[:2]: # Tampilkan maks 2
                st.markdown(f"**{alert[0]}**") 
                for detail in alert[1:3]: # Maks 2 detail
                    if detail: st.markdown(f"- {detail}")
                st.markdown("---")
        else:
            st.info("No critical alerts data currently.")
        # st.button("View All Alerts", type="primary", key="view_alerts") # Tombol bisa diarahkan ke halaman lain
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Predictive Hotspots")
        # hotspot_view = st.radio("View", ["Emerging", "Trending", "Predicted"], horizontal=True, key="hotspot_view")
        if predictive_hotspots_list:
            for hotspot in predictive_hotspots_list[:2]: # Tampilkan maks 2
                st.markdown(f"**{hotspot[0]}**")
                for detail in hotspot[1:3]: # Maks 2 detail
                    if detail: st.markdown(f"- {detail}")
                st.markdown("---")
        else:
            st.info("No predictive hotspots identified.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Prepare data for LLM context ---
    live_sentiment_summary_for_llm = {}
    if not sentiment_data_for_chart.empty:
        _total_sentiment = sentiment_data_for_chart['Value'].sum() if sentiment_data_for_chart['Value'].sum() > 0 else 1
        live_sentiment_summary_for_llm = {
            row['Category']: f"{(row['Value']/_total_sentiment*100):.1f}%"
            for index, row in sentiment_data_for_chart.iterrows()
        }

    live_intent_summary_for_llm = {}
    if not intent_data_for_chart.empty:
        _total_intent = intent_data_for_chart['Value'].sum() if intent_data_for_chart['Value'].sum() > 0 else 1
        live_intent_summary_for_llm = {
            row['Intent']: f"{(row['Value']/_total_intent*100):.1f}% (approx {row['Value']:.0f} mentions)"
            for index, row in intent_data_for_chart.iterrows()
        }

    live_volume_summary_for_llm = "Volume data N/A"
    if not vol_df_for_chart.empty:
        _volume_data_points = vol_df_for_chart['Volume'].tolist()
        live_volume_summary_for_llm = f"Volume trend over {len(_volume_data_points)} days: latest approx {int(_volume_data_points[-1]) if _volume_data_points else 'N/A'} interactions."


    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    # voice_view = st.radio("View", ["Overview", "Sentiment", "Intent", "Volume"], horizontal=True, key="voice_view") # Tombol view ini perlu implementasi lebih lanjut
    col1_snap, col2_snap, col3_snap = st.columns(3)

    with col1_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment Distribution")
        if not sentiment_data_for_chart.empty and sentiment_data_for_chart['Value'].sum() > 0:
            fig_sentiment = px.pie(sentiment_data_for_chart, values='Value', names='Category', color='Category',
                                   color_discrete_map={'Positif': '#34c759', 'Netral': '#a2a2a7', 'Negatif': '#ff3b30'}, hole=0.75)
            fig_sentiment.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=10)), showlegend=True)
            fig_sentiment.update_traces(textinfo='percent+label', textfont_size=10, insidetextorientation='radial')
            st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No sentiment data to display for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Intent Distribution")
        if not intent_data_for_chart.empty and intent_data_for_chart['Value'].sum() > 0:
            fig_intent = px.bar(intent_data_for_chart.head(5), y='Intent', x='Value', orientation='h', color='Intent', # Tampilkan top 5 intent
                                color_discrete_sequence=px.colors.qualitative.Pastel) # Peta warna bisa disesuaikan
            fig_intent.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None, showlegend=False)
            st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No intent data to display for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"### Volume Trend ({len(vol_df_for_chart)} Days)")
        if not vol_df_for_chart.empty and vol_df_for_chart['Volume'].sum() > 0:
            fig_volume = px.line(vol_df_for_chart, x='Day', y='Volume', line_shape='spline')
            fig_volume.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines')
            min_vol = vol_df_for_chart['Volume'].min() 
            max_vol = vol_df_for_chart['Volume'].max()
            fig_volume.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None,
                                     yaxis=dict(range=[min_vol - (max_vol-min_vol)*0.1 if max_vol > min_vol else min_vol-10, max_vol + (max_vol-min_vol)*0.1  if max_vol > min_vol else max_vol+10]))
            st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No volume data to display for the selected filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Summary text under voice snapshot
    if not sentiment_data_for_chart.empty and not intent_data_for_chart.empty:
        positive_percentage = live_sentiment_summary_for_llm.get('Positif','N/A')
        top_intent_name = intent_data_for_chart['Intent'].iloc[0] if not intent_data_for_chart.empty else "N/A"
        st.markdown(f"Positive sentiment leads at {positive_percentage}. '{top_intent_name}' is a top intent. Volume shows recent trends.")
    else:
        st.markdown("Summary data unavailable due to lack of interaction data for the current filter.")

    # Top Customer Themes
    st.markdown("## Top Customer Themes")
    # theme_view = st.radio("View", ["Top 10", "Trending", "Emerging", "Declining"], horizontal=True, key="theme_view")
    col1_theme, col2_theme = st.columns(2)

    with col1_theme:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Positive Themes")
        if positive_themes_list:
            for theme_item in positive_themes_list:
                st.markdown(f"- {theme_item[0][:100] + '...' if len(theme_item[0]) > 100 else theme_item[0]}") # Ringkas summary jika terlalu panjang
                # if len(theme_item) > 1 and theme_item[1]: st.markdown(f'> {theme_item[1][:150]}...') # Ringkas quote
        else:
            st.info("No prominent positive themes found for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_theme:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Negative Themes")
        if negative_themes_list:
            for theme_item in negative_themes_list:
                st.markdown(f"- {theme_item[0][:100] + '...' if len(theme_item[0]) > 100 else theme_item[0]}")
                # if len(theme_item) > 1 and theme_item[1]: st.markdown(f'> {theme_item[1][:150]}...')
        else:
            st.info("No prominent negative themes found for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Opportunity Radar
    st.markdown("## Opportunity Radar")
    # opportunity_view = st.radio("View", ["High Value", "Quick Wins", "Strategic"], horizontal=True, key="opportunity_view")
    if opportunity_radar_list:
        num_opportunities = len(opportunity_radar_list)
        cols_opportunity = st.columns(min(num_opportunities, 3)) 
        for i, opportunity in enumerate(opportunity_radar_list[:3]): # Tampilkan maks 3
            with cols_opportunity[i % 3]:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                category_icon = {"Delightful": "🎉", "Cross-Sell": "💰", "Service Excel": "⭐"}.get(opportunity[0], "💡")
                st.markdown(f"**{category_icon} {opportunity[0]}: {opportunity[1]}**") 
                if len(opportunity) > 2 and opportunity[2]: st.markdown(f"- {opportunity[2]}") 
                if len(opportunity) > 3 and opportunity[3]: st.markdown(f"- {opportunity[3]}") 
                if len(opportunity) > 4 and opportunity[4]: st.markdown(f"- Action: {opportunity[4]}") 
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No specific opportunities identified from current data or supplementary sheets.")


    # VIRA Chat Assistant
    st.markdown("## Chat with VIRA")
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm VIRA. Ask me about the dashboard insights."}]

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
                # Anda bisa tambahkan lebih banyak detail di sini jika VIRA perlu tahu
                # Misalnya, ringkasan tema, atau jumlah alert.
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
    st.write("This section is under development.")
