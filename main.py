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
SERVICE_ACCOUNT_FILE = 'key.json' # Pastikan file key.json ada di direktori yang sama atau sesuaikan path
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
        st.error(f"Service account key file '{SERVICE_ACCOUNT_FILE}' not found. Please ensure 'key.json' is in the correct location.")
        return None
    except RefreshError as e:
        st.error(f"Error with Google Sheets credentials: {e}. Check your 'key.json' and API permissions.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during Google Sheets authentication: {e}")
        return None

@st.cache_data(ttl=600)
def fetch_main_data(_service, sheet_name="Sheet1", range_name="A:H"): # ## MODIFIED ## Ganti ke Sheet1 atau nama sheet Anda
    if _service is None:
        return pd.DataFrame()
    try:
        result = _service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID,
            range=f"{sheet_name}!{range_name}"
        ).execute()
        values = result.get('values', [])
        if not values or len(values) < 1:
            st.warning(f"No data found in sheet '{sheet_name}'.")
            return pd.DataFrame()

        header = values[0]
        data_rows = values[1:]

        processed_rows = []
        num_cols = len(header)
        for row_idx, row in enumerate(data_rows):
            if len(row) < num_cols:
                processed_rows.append(row + [None] * (num_cols - len(row)))
            elif len(row) > num_cols:
                st.warning(f"Row {row_idx+2} in sheet '{sheet_name}' has more columns ({len(row)}) than header ({num_cols}). Truncating.")
                processed_rows.append(row[:num_cols])
            else:
                processed_rows.append(row)

        if not processed_rows:
            st.warning(f"No data rows found in sheet '{sheet_name}' after header.")
            return pd.DataFrame(columns=header) # Return empty DataFrame with headers

        df = pd.DataFrame(processed_rows, columns=header)

        # Konversi Tipe Data Penting
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            # Hapus baris dengan tanggal yang tidak valid jika diperlukan, tapi 'coerce' akan membuat NaT
            # df = df.dropna(subset=['Date']) 
        else:
            st.warning("Column 'Date' not found in the sheet. Time-based filtering and trends will not work.")

        for col in ['Sentimen', 'Intent', 'Product', 'Channel', 'Classification', 'Summary', 'masked_text']: ## MODIFIED ## Added Classification, Summary, masked_text
            if col in df.columns:
                df[col] = df[col].astype(str)
                if col == 'Sentimen':
                    df[col] = df[col].str.capitalize()
            elif col not in ['Summary', 'masked_text']: # Summary and masked_text might not be strictly required for all charts
                 st.warning(f"Expected column '{col}' not found in the sheet.")


        # Membersihkan nilai 'nan' string yang mungkin muncul dari konversi astype(str) pada kolom numerik kosong
        for col in df.columns:
            if df[col].dtype == 'object': # Hanya untuk kolom string
                df[col] = df[col].replace({'nan': None, 'None': None, '': None})


        return df
    except Exception as e:
        st.error(f"Could not fetch or process data from sheet '{sheet_name}'. Error: {e}")
        return pd.DataFrame()


# --- Data Processing Functions from Main DataFrame ---

def filter_dataframe(df, time_period_display, selected_products, selected_channels):
    if df.empty:
        return df

    df_filtered = df.copy() # Start with a copy

    # Filter Tanggal
    if 'Date' in df_filtered.columns and pd.api.types.is_datetime64_any_dtype(df_filtered['Date']):
        # Drop rows where Date is NaT before attempting to filter
        df_filtered = df_filtered.dropna(subset=['Date'])
        if df_filtered.empty:
            st.info("No valid dates found in data after initial load.")
            return pd.DataFrame(columns=df.columns) # Return empty with original columns

        end_date = df_filtered['Date'].max() 
        if pd.isna(end_date): # If all dates were NaT
             st.info("No valid dates available for filtering period.")
             return df # Return original if no valid end_date

        start_date_filter = None

        if time_period_display == "Today":
            start_date_filter = end_date.normalize() # Set to start of the max date
        elif time_period_display == "This Week": # Asumsi minggu dimulai Senin
            start_date_filter = (end_date - timedelta(days=end_date.weekday())).normalize()
        elif time_period_display == "This Month":
            start_date_filter = end_date.replace(day=1).normalize()
        elif time_period_display == "This Quarter":
            current_quarter = (end_date.month - 1) // 3 + 1
            first_month_of_quarter = 3 * (current_quarter - 1) + 1
            start_date_filter = end_date.replace(month=first_month_of_quarter, day=1).normalize()
        elif time_period_display == "This Year":
            start_date_filter = end_date.replace(month=1, day=1).normalize()
        # "All Periods" tidak perlu filter tanggal spesifik, jadi df_filtered tetap

        if start_date_filter:
             df_filtered = df_filtered[(df_filtered['Date'] >= start_date_filter) & (df_filtered['Date'] <= end_date)]
    elif 'Date' not in df_filtered.columns:
        st.warning("Date column missing, cannot filter by time period.")
    elif not pd.api.types.is_datetime64_any_dtype(df_filtered['Date']):
        st.warning("Date column is not in datetime format, cannot filter by time period.")


    # Filter Produk
    if 'Product' in df_filtered.columns:
        if "All Products" not in selected_products and selected_products:
            df_filtered = df_filtered[df_filtered['Product'].isin(selected_products)]
    else:
        st.warning("Product column missing, cannot filter by product.")

    # Filter Channel
    if 'Channel' in df_filtered.columns:
        if "All Channels" not in selected_channels and selected_channels:
            df_filtered = df_filtered[df_filtered['Channel'].isin(selected_channels)]
    else:
        st.warning("Channel column missing, cannot filter by channel.")

    return df_filtered


def get_sentiment_distribution(df):
    if df.empty or 'Sentimen' not in df.columns:
        return pd.DataFrame({'Category': ['Positif', 'Neutral', 'Negatif'], 'Value': [0, 0, 0]})
    # Pastikan hanya nilai yang valid (Positif, Netral, Negatif) yang dihitung
    valid_sentiments = ['Positif', 'Netral', 'Negatif']
    df_valid_sentiment = df[df['Sentimen'].isin(valid_sentiments)]
    sentiment_counts = df_valid_sentiment['Sentimen'].value_counts().reindex(valid_sentiments, fill_value=0)
    return pd.DataFrame({'Category': sentiment_counts.index, 'Value': sentiment_counts.values})

def get_intent_distribution(df):
    if df.empty or 'Intent' not in df.columns:
        return pd.DataFrame({'Intent': [], 'Value': []})
    intent_counts = df['Intent'].value_counts().nlargest(10) # Ambil top 10 intent
    return pd.DataFrame({'Intent': intent_counts.index, 'Value': intent_counts.values})

## ADDED ##
def get_classification_distribution(df):
    if df.empty or 'Classification' not in df.columns:
        return pd.DataFrame({'Classification': [], 'Value': []})
    classification_counts = df['Classification'].value_counts().nlargest(10) # Ambil top 10
    return pd.DataFrame({'Classification': classification_counts.index, 'Value': classification_counts.values})

def get_volume_trend(df, days=30):
    if df.empty or 'Date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Date']):
        return pd.DataFrame({'Day_Date': [], 'Day': [], 'Volume': []}) # ## MODIFIED ## Added Day_Date for x-axis label

    df_dated = df.dropna(subset=['Date']) # Pastikan tidak ada NaT
    if df_dated.empty:
        return pd.DataFrame({'Day_Date': [], 'Day': [], 'Volume': []})

    max_date_in_data = df_dated['Date'].max()
    if pd.isna(max_date_in_data): # jika semua tanggal NaT
        return pd.DataFrame({'Day_Date': [], 'Day': [], 'Volume': []})

    start_date = max_date_in_data - timedelta(days=days-1)
    start_date = start_date.normalize() # Pastikan mulai dari awal hari

    volume_data = df_dated[df_dated['Date'] >= start_date].groupby(df_dated['Date'].dt.date)['Date'].count().reset_index(name='Volume')
    volume_data.rename(columns={'Date': 'Day_Date'}, inplace=True)

    all_days_dt = pd.date_range(start=start_date, end=max_date_in_data, freq='D')
    all_days_df = pd.DataFrame({'Day_Date': all_days_dt.date})


    volume_data_full = pd.merge(all_days_df, volume_data, on='Day_Date', how='left').fillna(0)
    volume_data_full['Day_Date'] = pd.to_datetime(volume_data_full['Day_Date']) # Pastikan datetime untuk formatting
    volume_data_full['Day'] = volume_data_full['Day_Date'].dt.strftime('%b %d') # Label hari untuk sumbu x yang lebih baik

    return volume_data_full[['Day_Date', 'Day', 'Volume']]


def get_top_themes(df, sentiment_filter, top_n=3):
    if df.empty or 'Summary' not in df.columns or 'Sentimen' not in df.columns:
        return []

    # Filter baris yang 'Summary' atau 'masked_text' nya valid (bukan None atau string kosong)
    filtered_df = df[
        (df['Sentimen'] == sentiment_filter) &
        (df['Summary'].notna()) & (df['Summary'] != '') &
        (df['masked_text'].notna()) & (df['masked_text'] != '')
    ].copy() # Use .copy() to avoid SettingWithCopyWarning

    if filtered_df.empty:
        return []

    theme_counts = filtered_df['Summary'].value_counts().nlargest(top_n)

    themes_list = []
    for summary, count in theme_counts.items():
        # Ambil contoh masked_text yang sesuai dengan summary tersebut
        # Pastikan ada, dan ambil yang pertama jika ada beberapa
        example_texts = filtered_df[filtered_df['Summary'] == summary]['masked_text']
        example_masked_text = example_texts.iloc[0] if not example_texts.empty else "No example quote available."

        # Ekstrak bagian inti dari masked_text, misal setelah __cus__
        if "__cus__" in example_masked_text:
            quote_start_index = example_masked_text.find("__cus__") + len("__cus__")
            # Ambil hingga 100 karakter pertama setelah __cus__ atau hingga __adm__ jika ada
            quote_end_index_adm = example_masked_text.find("__adm__", quote_start_index)
            if quote_end_index_adm != -1:
                 actual_quote = example_masked_text[quote_start_index:quote_end_index_adm].strip()
            else:
                actual_quote = example_masked_text[quote_start_index:].strip()

            actual_quote = actual_quote.splitlines()[0] if actual_quote else "" # Ambil baris pertama saja
            actual_quote = (actual_quote[:70] + '...') if len(actual_quote) > 70 else actual_quote
        else:
            actual_quote = (example_masked_text[:70] + '...') if len(example_masked_text) > 70 else example_masked_text


        themes_list.append([summary, f'"{actual_quote}"'])
    return themes_list


def calculate_health_score_data(df, time_period_key="month"):
    if df.empty:
        return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "Data unavailable"
        }

    sentiment_map = {'Positif': 1, 'Netral': 0.5, 'Negatif': 0}
    if 'Sentimen' not in df.columns:
        st.warning("Health Score: 'Sentimen' column missing.")
        return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "Sentimen data missing"
        }
    df['SentimentScore'] = df['Sentimen'].map(sentiment_map).fillna(0.5)

    if 'Date' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['Date']):
         st.warning("Health Score: 'Date' column missing or not datetime.")
         return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "Date column missing/invalid"
        }

    df_dated = df.dropna(subset=['Date'])
    if df_dated.empty:
        return { "labels": ["N/A"], "values": [0], "score": 0, "trend": "N/A", "trend_positive": False, "trend_label": "No valid dates for health score"}


    daily_scores = df_dated.groupby(df_dated['Date'].dt.date)['SentimentScore'].mean().reset_index()
    daily_scores['SentimentScore'] = daily_scores['SentimentScore'] * 100
    daily_scores.rename(columns={'Date': 'Day_Date'}, inplace=True) # ## MODIFIED ##
    daily_scores['Day_Date'] = pd.to_datetime(daily_scores['Day_Date']) # ## MODIFIED ## ensure datetime for sorting

    if daily_scores.empty:
         return {
            "labels": ["N/A"], "values": [0], "score": 0,
            "trend": "N/A", "trend_positive": False, "trend_label": "No daily scores"
        }

    daily_scores = daily_scores.sort_values(by='Day_Date') # ## ADDED ## Ensure sorted by date

    current_score = daily_scores['SentimentScore'].iloc[-1] if not daily_scores.empty else 0
    previous_score = daily_scores['SentimentScore'].iloc[-2] if len(daily_scores) > 1 else current_score

    trend_value = current_score - previous_score
    trend_percentage_str = f"{trend_value:+.1f}%" # No need to check for previous_score !=0 due to :+

    num_points_to_show = 30 # For trend line in health score
    trend_data_points = daily_scores.tail(num_points_to_show)
    trend_labels = [d.strftime('%b %d') for d in trend_data_points['Day_Date']]
    trend_values = trend_data_points['SentimentScore'].tolist()


    if not trend_labels:
        trend_labels = ["N/A"]
        trend_values = [0]
        current_score = 0

    return {
        "labels": trend_labels,
        "values": trend_values,
        "score": int(round(current_score)),
        "trend": trend_percentage_str,
        "trend_positive": trend_value >= 0,
        "trend_label": "vs. previous period",
    }


@st.cache_data(ttl=600)
def fetch_supplementary_data(_service, sheet_name, range_name, num_cols=1):
    if _service is None: return []
    try:
        result = _service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=f"{sheet_name}!{range_name}").execute()
        values = result.get('values', [])
        if not values: return [] # No data at all

        # Check if only header exists or header + no data
        if len(values) < 1: return [] # No header
        if len(values) == 1 and not any(values[0]): return [] # Empty header
        if len(values) == 1 and any(values[0]): return [] # Only header, no data rows

        items = []
        for row in values[1:]: # Skip header
            if row and any(r.strip() for r in row if isinstance(r, str)): # Check if row is not empty and has non-whitespace content
                 items.append([row[i] if i < len(row) else "" for i in range(num_cols)])
        return items
    except Exception as e:
        st.warning(f"Could not fetch data from supplementary sheet '{sheet_name}'. Error: {e}")
        return []

# Initialize Google Sheets Service
sheets_service = get_google_sheets_service()

# Set page configuration
st.set_page_config(page_title="Voice of Customer Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    /* General App Styling */
    .stApp {
        /* background-color: #f0f2f5; */ /* Light grey background */
    }
    /* Metric Card Styling */
    .metric-card {
        background-color: #ffffff;
        padding: 15px 20px; /* Reduced padding */
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 15px; /* Reduced margin */
        height: 330px; /* Fixed height for cards */
        display: flex;
        flex-direction: column;
    }
    .metric-card h3 {
        font-size: 1.1em; /* Smaller title */
        margin-top: 0;
        margin-bottom: 8px; /* Reduced margin */
        color: #333;
    }
    .metric-card .stPlotlyChart {
        margin-top: auto; /* Push chart to bottom if content is less */
    }

    /* Health Score Specific */
    .metric-value {
        font-size: 2.5em; /* Smaller main score value */
        font-weight: bold;
        color: #1a1a1a;
        margin-bottom: 0px;
    }
    .metric-trend-positive {
        color: #34c759; /* Green for positive trend */
        font-size: 0.9em;
    }
    .metric-trend-negative {
        color: #ff3b30; /* Red for negative trend */
        font-size: 0.9em;
    }
    .stRadio > label {
        font-size: 0.8em !important; /* Smaller radio button labels */
        padding: 2px 5px !important; /* Tighter radio buttons */
    }
    .stRadio div[role="radiogroup"] > label { /* Target individual radio items */
        margin-right: 5px !important;
    }
    /* Alert/Hotspot list styling */
    .metric-card ul {
        padding-left: 20px;
        font-size: 0.9em;
        margin-bottom: 5px;
    }
    .metric-card li {
        margin-bottom: 3px;
    }
    .metric-card hr {
        margin-top: 5px;
        margin-bottom: 5px;
        border-top: 1px solid #eee;
    }
    /* Chat input */
    .stChatInputContainer > div > div > textarea {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


# --- NVIDIA API Client Initialization ---
SYSTEM_PROMPT_VIRA = """Anda adalah VIRA, asisten AI analitik CX yang cerdas dan membantu.
Anda memiliki akses ke ringkasan data dasbor Voice of Customer (VOC) saat ini.
Tugas Anda adalah menjawab pertanyaan pengguna tentang data ini, memberikan wawasan, dan membantu mereka memahami sentimen pelanggan, niat, tren volume, dan tema utama.
Gunakan data yang diberikan dalam ringkasan dasbor untuk menjawab pertanyaan secara akurat.
Jika pertanyaan berada di luar cakupan data yang diberikan, nyatakan bahwa informasi tersebut tidak tersedia di dasbor saat ini.
Bersikaplah ringkas, profesional, dan berorientasi pada solusi.
Data yang tersedia:
- Skor Kesehatan Pelanggan (persentase, tren naik/turun, dan label periode pembanding).
- Distribusi Sentimen (persentase Positif, Netral, Negatif).
- Distribusi Niat (persentase dan jumlah sebutan untuk niat utama).
- Distribusi Klasifikasi (persentase dan jumlah sebutan untuk klasifikasi utama).
- Tren Volume (deskripsi tren volume selama periode tertentu).
- Tema Positif Teratas (daftar tema positif utama dengan contoh kutipan).
- Tema Negatif Teratas (daftar tema negatif utama dengan contoh kutipan).
Mulai respons Anda dengan ramah dan langsung ke intinya. Jangan mengulang pertanyaan pengguna secara verbatim.
Hindari membuat asumsi atau memberikan informasi yang tidak didukung oleh data dasbor.
"""
NVIDIA_API_KEY = st.secrets.get("NVIDIA_API_KEY") # Removed default key
client = None

if not NVIDIA_API_KEY:
    st.sidebar.warning("NVIDIA API Key not configured in secrets. VIRA chat will be disabled.")
else:
    try:
        client = OpenAI(
          base_url = "https://integrate.api.nvidia.com/v1",
          api_key = NVIDIA_API_KEY
        )
    except Exception as e:
        st.error(f"Error initializing OpenAI (NVIDIA) client: {e}")


def generate_llm_response(user_prompt: str, dashboard_state: dict, system_prompt: str):
    if client is None:
        yield "Layanan AI (VIRA) tidak dikonfigurasi atau tidak tersedia saat ini."
        return

    sentiment_summary_str = ', '.join([f"{k}: {v}" for k, v in dashboard_state.get('sentiment_summary', {}).items()])
    intent_summary_str = ('; '.join([f"{k}: {v}" for k, v in dashboard_state.get('intent_summary', {}).items()])
                          if dashboard_state.get('intent_summary') else 'N/A')
    classification_summary_str = ('; '.join([f"{k}: {v}" for k, v in dashboard_state.get('classification_summary', {}).items()])
                                  if dashboard_state.get('classification_summary') else 'N/A') ## ADDED ##

    positive_themes_str = "\n".join([f"- {theme[0]}: {theme[1]}" for theme in dashboard_state.get('positive_themes', [])])
    negative_themes_str = "\n".join([f"- {theme[0]}: {theme[1]}" for theme in dashboard_state.get('negative_themes', [])])


    dashboard_summary_for_llm = f"""
Ringkasan Tampilan Dasbor Saat Ini (berdasarkan filter yang dipilih pengguna):
- Periode Waktu Terpilih: {dashboard_state.get('time_period_label', 'N/A')}
- Skor Kesehatan Pelanggan: {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')} {dashboard_state.get('trend_label', 'N/A')})
- Distribusi Sentimen: {sentiment_summary_str if sentiment_summary_str else 'N/A'}.
- Distribusi Niat Utama: {intent_summary_str}.
- Distribusi Klasifikasi Utama: {classification_summary_str}.
- Tren Volume: {dashboard_state.get('volume_summary', 'Data volume N/A')}.
- Tema Positif Teratas:
{positive_themes_str if positive_themes_str else "  Tidak ada tema positif yang dominan untuk filter saat ini."}
- Tema Negatif Teratas:
{negative_themes_str if negative_themes_str else "  Tidak ada tema negatif yang dominan untuk filter saat ini."}
"""
    constructed_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{dashboard_summary_for_llm}\n\nPertanyaan Pengguna: \"{user_prompt}\""}
    ]
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama3-chatqa-1.5-8b", # More general purpose model or a newer one if available
            messages=constructed_messages,
            temperature=0.2, # Lower temperature for more factual responses based on context
            top_p=0.8,
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        error_message = f"Maaf, terjadi kesalahan saat menghubungi layanan AI: {str(e)}."
        print(f"LLM API Error: {e}") # For server-side logging
        yield error_message

# Sidebar
with st.sidebar:
    st.title("VOCAL")
    st.markdown("CX Analytics Platform")
    st.markdown("---")
    # st.header("Menu") # Can be removed if selectbox is clear enough
    # if "menu_nav" not in st.session_state:
    #     st.session_state.menu_nav = "Dashboard"
    # page = st.selectbox("Navigation", ["Dashboard", "Analytics", "Feedback", "Alerts", "Reports"], key="menu_nav", label_visibility="collapsed")
    page = "Dashboard" # For this single page app version

    st.markdown("---")
    st.markdown("Powered by **NVIDIA NIM** & **Streamlit**")
    st.markdown("---")
    st.markdown("**Sebastian**")
    st.markdown("CX Manager")


# Fetch Main Data Once
main_df_raw = pd.DataFrame()
if sheets_service:
    main_df_raw = fetch_main_data(sheets_service, sheet_name="Sheet1") # ## MODIFIED ## Pastikan nama sheet benar
    if main_df_raw.empty:
        st.warning("No data returned from Google Sheets. Please check the sheet name, range, and content.")
    # else: # Optionally print some info for debugging
        # st.sidebar.metric("Total Rows Loaded", len(main_df_raw))
        # st.sidebar.write("Columns:", main_df_raw.columns.tolist())
else:
    st.error("Google Sheets service not available. Cannot load main data.")
    # Fallback: Load from CSV if sheets_service fails (optional)
    # try:
    #     main_df_raw = pd.read_csv("fallback_main_data.csv") # you'd need this file
    #     st.info("Loaded data from fallback_main_data.csv")
    # except FileNotFoundError:
    #     st.error("Fallback CSV not found.")

if main_df_raw.empty and sheets_service:
    st.warning("Failed to load data from Google Sheets or data is empty. Dashboard might be empty or show errors.")


# Main content
if page == "Dashboard":
    st.title("Customer Experience Health")
    st.markdown("Real-time Insights & Performance Overview")

    # Filters
    filter_cols = st.columns([2,2,2,1]) # Add a spacer column
    with filter_cols[0]:
        time_period_display = st.selectbox(
            "TIME PERIOD",
            ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
            index=0, key="time_filter" # Default to "All Periods"
        )

    unique_products = ["All Products"]
    if not main_df_raw.empty and 'Product' in main_df_raw.columns:
        # Filter out None or empty strings before getting unique values
        valid_products = main_df_raw['Product'].dropna().astype(str).unique()
        unique_products.extend(sorted([p for p in valid_products if p and p.strip()]))

    unique_channels = ["All Channels"]
    if not main_df_raw.empty and 'Channel' in main_df_raw.columns:
        valid_channels = main_df_raw['Channel'].dropna().astype(str).unique()
        unique_channels.extend(sorted([c for c in valid_channels if c and c.strip()]))

    with filter_cols[1]:
        products_filter = st.multiselect("PRODUCT", unique_products, default=["All Products"], key="product_filter")
    with filter_cols[2]:
        channels_filter = st.multiselect("CHANNEL", unique_channels, default=["All Channels"], key="channel_filter")

    # Filter Data Utama berdasarkan pilihan filter
    df_filtered = pd.DataFrame(columns=main_df_raw.columns) # Initialize with columns
    if not main_df_raw.empty:
        df_filtered = filter_dataframe(main_df_raw, time_period_display, products_filter, channels_filter)
        if df_filtered.empty and not main_df_raw.empty : # If filtering resulted in empty
             st.info(f"No data available for the selected filters: {time_period_display}, Products: {products_filter}, Channels: {channels_filter}")
    elif main_df_raw.empty and sheets_service:
        st.error("Source data is empty. Cannot apply filters or display charts.")
    # else: # sheets_service is None, error already shown

    # --- AGREGASI DATA DARI df_filtered ---
    sentiment_data_for_chart = get_sentiment_distribution(df_filtered)
    intent_data_for_chart = get_intent_distribution(df_filtered)
    classification_data_for_chart = get_classification_distribution(df_filtered) ## ADDED ##
    vol_df_for_chart = get_volume_trend(df_filtered, days=30)

    current_health_data = calculate_health_score_data(df_filtered, time_period_key=time_period_display.lower().replace("this ",""))
    current_health_data['time_period_label'] = time_period_display

    positive_themes_list = get_top_themes(df_filtered, "Positif", top_n=3)
    negative_themes_list = get_top_themes(df_filtered, "Negatif", top_n=3)

    critical_alerts_list = []
    predictive_hotspots_list = []
    opportunity_radar_list = []
    if sheets_service:
        # These supplementary sheets are optional.
        critical_alerts_list = fetch_supplementary_data(sheets_service, "CriticalAlertsData", "A:D", num_cols=4)
        predictive_hotspots_list = fetch_supplementary_data(sheets_service, "PredictiveHotspotsData", "A:D", num_cols=4)
        opportunity_radar_list = fetch_supplementary_data(sheets_service, "OpportunityRadarData", "A:E", num_cols=5)
    # else: # No need for fallback if sheets_service is None, already handled


    # --- Dashboard widgets ---
    # st.markdown("## Dashboard Widgets") # Section title, can be removed for cleaner look
    col1_dash, col2_dash, col3_dash = st.columns(3)

    with col1_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Customer Health Score")
        score_col1, score_col2 = st.columns([1, 2])
        with score_col1:
            st.markdown(f'<div class="metric-value">{current_health_data.get("score", "N/A")}%</div>', unsafe_allow_html=True)
        with score_col2:
            trend_positive = current_health_data.get("trend_positive", False)
            trend_icon = "↑" if trend_positive else "↓"
            trend_class = "metric-trend-positive" if trend_positive else "metric-trend-negative"
            st.markdown(f'<div class="{trend_class}" style="margin-top:10px;">{trend_icon} {current_health_data.get("trend", "N/A")} {current_health_data.get("trend_label", "")}</div>', unsafe_allow_html=True)

        fig_health = go.Figure()
        if current_health_data.get("values") and any(v is not None for v in current_health_data.get("values")): # Check for non-empty and non-None values
            min_val_health = min(v for v in current_health_data.get("values", [0]) if v is not None)
            max_val_health = max(v for v in current_health_data.get("values", [100]) if v is not None)

            fig_health.add_trace(go.Scatter(
                x=current_health_data.get("labels", []),
                y=current_health_data.get("values", []),
                mode='lines+markers', fill='tozeroy', fillcolor='rgba(52,199,89,0.1)',
                line=dict(color='#34c759', width=2), name='Health Score',
                marker=dict(size=4)
            ))
            fig_health.update_layout(
                height=150, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9)),
                yaxis=dict(showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9), range=[max(0, min_val_health - 5), min(100, max_val_health + 5)])
            )
        st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})
        st.markdown("<p style='font-size:0.8em; color:#666; margin-top:5px;'>Customer health based on sentiment analysis of interactions.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Critical Alerts")
        if critical_alerts_list:
            for alert_idx, alert in enumerate(critical_alerts_list[:3]): # Show max 3
                st.markdown(f"**{alert[0]}**")
                if len(alert) > 1 and alert[1]: st.markdown(f"<p style='font-size:0.85em; color:#555;'>- {alert[1]}</p>", unsafe_allow_html=True)
                if len(alert) > 2 and alert[2]: st.markdown(f"<p style='font-size:0.85em; color:#555;'>- {alert[2]}</p>", unsafe_allow_html=True)
                if alert_idx < len(critical_alerts_list[:3]) - 1 : st.markdown("---") # Add separator if not the last item
        else:
            st.info("No critical alerts data currently.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3_dash:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Predictive Hotspots")
        if predictive_hotspots_list:
            for hotspot_idx, hotspot in enumerate(predictive_hotspots_list[:3]): # Show max 3
                st.markdown(f"**{hotspot[0]}**")
                if len(hotspot) > 1 and hotspot[1]: st.markdown(f"<p style='font-size:0.85em; color:#555;'>- {hotspot[1]}</p>", unsafe_allow_html=True)
                if len(hotspot) > 2 and hotspot[2]: st.markdown(f"<p style='font-size:0.85em; color:#555;'>- {hotspot[2]}</p>", unsafe_allow_html=True)
                if hotspot_idx < len(predictive_hotspots_list[:3]) - 1 : st.markdown("---")
        else:
            st.info("No predictive hotspots identified.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- Prepare data for LLM context ---
    live_sentiment_summary_for_llm = {}
    if not sentiment_data_for_chart.empty and sentiment_data_for_chart['Value'].sum() > 0:
        _total_sentiment = sentiment_data_for_chart['Value'].sum()
        live_sentiment_summary_for_llm = {
            row['Category']: f"{(row['Value']/_total_sentiment*100):.1f}% ({row['Value']:.0f})"
            for index, row in sentiment_data_for_chart.iterrows()
        }

    live_intent_summary_for_llm = {}
    if not intent_data_for_chart.empty and intent_data_for_chart['Value'].sum() > 0:
        _total_intent = intent_data_for_chart['Value'].sum()
        live_intent_summary_for_llm = {
            row['Intent']: f"{(row['Value']/_total_intent*100):.1f}% ({row['Value']:.0f})"
            for index, row in intent_data_for_chart.iterrows()
        }

    ## ADDED ##
    live_classification_summary_for_llm = {}
    if not classification_data_for_chart.empty and classification_data_for_chart['Value'].sum() > 0:
        _total_classification = classification_data_for_chart['Value'].sum()
        live_classification_summary_for_llm = {
            row['Classification']: f"{(row['Value']/_total_classification*100):.1f}% ({row['Value']:.0f})"
            for index, row in classification_data_for_chart.iterrows()
        }

    live_volume_summary_for_llm = "Volume data N/A"
    if not vol_df_for_chart.empty and 'Volume' in vol_df_for_chart.columns:
        _volume_data_points = vol_df_for_chart['Volume'].tolist()
        if _volume_data_points:
             live_volume_summary_for_llm = f"Tren volume selama {len(_volume_data_points)} hari terakhir (data terbaru sekitar {int(_volume_data_points[-1])} interaksi)."


    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    col1_snap, col2_snap, col3_snap, col4_snap = st.columns(4) ## MODIFIED ## Added 4th column for Classification

    with col1_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment")
        if not sentiment_data_for_chart.empty and sentiment_data_for_chart['Value'].sum() > 0:
            fig_sentiment = px.pie(sentiment_data_for_chart, values='Value', names='Category', color='Category',
                                   color_discrete_map={'Positif': '#34c759', 'Netral': '#a2a2a7', 'Negatif': '#ff3b30'}, hole=0.65)
            fig_sentiment.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
                                        plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=9)), showlegend=True)
            fig_sentiment.update_traces(textinfo='percent', textfont_size=10, insidetextorientation='radial') #label removed from inside
            st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No sentiment data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Intents")
        if not intent_data_for_chart.empty and intent_data_for_chart['Value'].sum() > 0:
            fig_intent = px.bar(intent_data_for_chart.head(5), y='Intent', x='Value', orientation='h',
                                color_discrete_sequence=['#007aff']*len(intent_data_for_chart.head(5))) # Single color
            fig_intent.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None, showlegend=False,
                                     yaxis=dict(autorange="reversed", tickfont=dict(size=9)), # Show top intent at the top
                                     xaxis=dict(tickfont=dict(size=9)))
            fig_intent.update_traces(texttemplate='%{x}', textposition='outside', textfont_size=9)
            st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No intent data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    ## ADDED ##
    with col3_snap:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Top Classifications")
        if not classification_data_for_chart.empty and classification_data_for_chart['Value'].sum() > 0:
            fig_classification = px.bar(classification_data_for_chart.head(5), y='Classification', x='Value', orientation='h',
                                        color_discrete_sequence=['#5856d6']*len(classification_data_for_chart.head(5))) # Purple
            fig_classification.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
                                             plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None, showlegend=False,
                                             yaxis=dict(autorange="reversed", tickfont=dict(size=9)),
                                             xaxis=dict(tickfont=dict(size=9)))
            fig_classification.update_traces(texttemplate='%{x}', textposition='outside', textfont_size=9)
            st.plotly_chart(fig_classification, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No classification data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4_snap: ## MODIFIED ## was col3_snap
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f"### Volume Trend") # Removed days from title, implied by x-axis
        if not vol_df_for_chart.empty and 'Volume' in vol_df_for_chart.columns and vol_df_for_chart['Volume'].sum() > 0:
            fig_volume = px.area(vol_df_for_chart, x='Day', y='Volume', line_shape='spline') # Use Day (formatted date) for x-axis
            fig_volume.update_traces(line_color='#ff9500', fillcolor='rgba(255,149,0,0.1)') # Orange color
            min_vol = vol_df_for_chart['Volume'].min()
            max_vol = vol_df_for_chart['Volume'].max()
            padding = (max_vol - min_vol) * 0.1 if max_vol > min_vol else 5 # Dynamic padding
            fig_volume.update_layout(height=230, margin=dict(l=0, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)',
                                     plot_bgcolor='rgba(0,0,0,0)', xaxis_title=None, yaxis_title=None,
                                     yaxis=dict(range=[max(0, min_vol - padding), max_vol + padding], tickfont=dict(size=9)),
                                     xaxis=dict(tickfont=dict(size=9), tickangle=-45))
            st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No volume data for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)


    # Top Customer Themes
    st.markdown("## Top Customer Themes")
    col1_theme, col2_theme = st.columns(2)

    with col1_theme:
        st.markdown('<div class="metric-card" style="height: auto; min-height: 330px;">', unsafe_allow_html=True) # Auto height for themes
        st.markdown("### <span style='color:#34c759'>●</span> Top Positive Themes", unsafe_allow_html=True)
        if positive_themes_list:
            for theme_item in positive_themes_list:
                st.markdown(f"**{theme_item[0][:100] + ('...' if len(theme_item[0]) > 100 else '')}**")
                if len(theme_item) > 1 and theme_item[1]:
                    st.markdown(f"<p style='font-size:0.85em; color:#555; margin-left: 10px;'><em>{theme_item[1]}</em></p>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No prominent positive themes found for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2_theme:
        st.markdown('<div class="metric-card" style="height: auto; min-height: 330px;">', unsafe_allow_html=True) # Auto height
        st.markdown("### <span style='color:#ff3b30'>●</span> Top Negative Themes", unsafe_allow_html=True)
        if negative_themes_list:
            for theme_item in negative_themes_list:
                st.markdown(f"**{theme_item[0][:100] + ('...' if len(theme_item[0]) > 100 else '')}**")
                if len(theme_item) > 1 and theme_item[1]:
                    st.markdown(f"<p style='font-size:0.85em; color:#555; margin-left: 10px;'><em>{theme_item[1]}</em></p>", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.info("No prominent negative themes found for current filters.")
        st.markdown('</div>', unsafe_allow_html=True)

    # Opportunity Radar
    st.markdown("## Opportunity Radar")
    if opportunity_radar_list:
        num_opportunities = len(opportunity_radar_list)
        cols_opportunity = st.columns(min(num_opportunities, 3))
        for i, opportunity in enumerate(opportunity_radar_list[:min(num_opportunities, 3)]): # Display up to 3
            with cols_opportunity[i % 3]:
                st.markdown('<div class="metric-card" style="height: auto; min-height:150px;">', unsafe_allow_html=True) # Auto height
                category_icon = {"Delightful": "🎉", "Cross-Sell": "💰", "Service Excel": "⭐", "Product Enhancement": "🚀", "Process Improvement": "🛠️"}.get(opportunity[0], "💡")
                st.markdown(f"**{category_icon} {opportunity[0]}**")
                if len(opportunity) > 1 and opportunity[1]: st.markdown(f"<p style='font-size:0.9em;'><em>{opportunity[1]}</em></p>", unsafe_allow_html=True)
                if len(opportunity) > 2 and opportunity[2]: st.markdown(f"<p style='font-size:0.8em; color:#555;'>- Detail: {opportunity[2]}</p>", unsafe_allow_html=True)
                if len(opportunity) > 3 and opportunity[3]: st.markdown(f"<p style='font-size:0.8em; color:#555;'>- Impact: {opportunity[3]}</p>", unsafe_allow_html=True)
                if len(opportunity) > 4 and opportunity[4]: st.markdown(f"<p style='font-size:0.8em; color:#555;'>- Action: {opportunity[4]}</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No specific opportunities identified from current data or supplementary sheets.")


    # VIRA Chat Assistant
    st.markdown("---")
    st.markdown("## Chat with VIRA (AI CX Analyst)")
    if client: # Only show chat if client is initialized
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm VIRA. Ask me about the dashboard insights based on the current filters."}]

        for message in st.session_state.messages[-10:]: # Show last 10 messages
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask VIRA (e.g., 'What are the main negative themes?')"):
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
                    "classification_summary": live_classification_summary_for_llm, ## ADDED ##
                    "volume_summary": live_volume_summary_for_llm,
                    "positive_themes": positive_themes_list,
                    "negative_themes": negative_themes_list,
                }
                try:
                    for chunk in generate_llm_response(prompt, dashboard_state_for_llm, SYSTEM_PROMPT_VIRA):
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e:
                    full_response = f"An unexpected error occurred with VIRA: {str(e)}"
                    st.error(full_response) # Show error in chat too
                    # Log this error on the server for debugging
                    print(f"Error during LLM stream: {e}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.warning("VIRA AI Assistant is currently unavailable. Please check API key configuration.")

else: # Other pages (not used in this single-page version)
    st.markdown(f"## {page}")
    st.write("This section is under development.")
