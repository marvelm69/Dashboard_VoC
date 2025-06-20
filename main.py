# -*- coding: utf-8 -*-
"""
main.py

A Streamlit web application for a Voice of Customer (VOC) Dashboard.

This application visualizes customer interaction data from various channels,
displaying metrics on customer health, sentiment, intent, and volume. It features
interactive filters and an AI-powered chat assistant (VIRA) to provide
data-driven insights.
"""

# --- IMPORTS ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from openai import OpenAI
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build

# --- CONFIGURATION ---

# Streamlit Page Configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Google Sheets API Configuration
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = '1V5cRgnnN5GTFsD9bR05hLzsKRWkhdEy3LhuTvSnUyIM' # Replace with your actual Spreadsheet ID
RANGE_NAME = 'sheet1!A:H'

# NVIDIA API Configuration
NVIDIA_API_KEY = "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" # Replace with your NVIDIA API key or use st.secrets
SYSTEM_PROMPT_VIRA = """
[span_0](start_span)Anda adalah VIRA, seorang konsultan virtual untuk Bank BCA.[span_0](end_span)
[span_1](start_span)Tugas utama Anda adalah menganalisis data dasbor yang disediakan dan memberikan wawasan, ringkasan, serta saran yang relevan.[span_1](end_span)
[span_2](start_span)Fokuslah pada metrik seperti skor kesehatan (jika ada), tren, sentimen pelanggan, niat panggilan, dan volume panggilan berdasarkan data yang disaring.[span_2](end_span)
[span_3](start_span)Selalu dasarkan jawaban Anda pada data yang diberikan dalam `dashboard_state`.[span_3](end_span)
[span_4](start_span)Gunakan bahasa Indonesia yang sopan dan mudah dimengerti.[span_4](end_span)
[span_5](start_span)Jika ada pertanyaan yang tidak dapat dijawab dari data dasbor, sampaikan dengan sopan bahwa informasi tersebut tidak tersedia dalam tampilan dasbor saat ini atau minta pengguna untuk memberikan detail lebih lanjut.[span_5](end_span)
[span_6](start_span)Berikan analisis yang ringkas namun mendalam.[span_6](end_span)
Jika ada pertanyaan yang diluar konteks analisis anda, sampaikan bahwa itu diluar kapabilitas anda untuk menjelaskannya.
PENTING:
Sebelum memberikan jawaban akhir kepada pengguna, Anda BOLEH melakukan analisis internal atau "berpikir".
[span_7](start_span)Jika Anda melakukan proses berpikir internal, *JANGAN* tuliskan pemikiran tersebut.[span_7](end_span)
[span_8](start_span)Jika tidak ada proses berpikir khusus atau analisis internal yang perlu dituliskan, langsung berikan jawaban.[span_8](end_span)
"""

# --- CUSTOM STYLING (CSS) ---
st.markdown("""
<style>
    .stApp {
        background-color: #f5f5f7;
        [span_9](start_span)color: #1d1d1f;[span_9](end_span)
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        padding: 20px;
        [span_10](start_span)border-radius: 10px;[span_10](end_span)
        [span_11](start_span)box-shadow: 0 2px 5px rgba(0,0,0,0.1);[span_11](end_span)
    }
    .stButton>button {
        background-color: #007aff;
        [span_12](start_span)color: white;[span_12](end_span)
        border-radius: 8px;
        border: none;
        [span_13](start_span)padding: 8px 16px;[span_13](end_span)
    }
    .stButton>button:hover {
        [span_14](start_span)background-color: #005bb5;[span_14](end_span)
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        [span_15](start_span)border-radius: 10px;[span_15](end_span)
        [span_16](start_span)box-shadow: 0 2px 5px rgba(0,0,0,0.1);[span_16](end_span)
        margin-bottom: 20px;
        height: 100%;
    }
    .metric-title {
        font-size: 18px;
        [span_17](start_span)font-weight: bold;[span_17](end_span)
        [span_18](start_span)color: #1d1d1f;[span_18](end_span)
    }
    .metric-value {
        font-size: 36px;
        [span_19](start_span)font-weight: bold;[span_19](end_span)
        color: #1d1d1f;
    }
    .metric-trend-positive {
        [span_20](start_span)color: #34c759;[span_20](end_span)
        [span_21](start_span)font-size: 16px;[span_21](end_span)
        font-weight: 500;
    }
    .metric-trend-negative {
        [span_22](start_span)color: #ff3b30;[span_22](end_span)
        [span_23](start_span)font-size: 16px;[span_23](end_span)
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


# --- DATA LOADING ---

@st.cache_data(ttl=600)
def load_data_from_google_sheets():
    """
    Fetches and preprocesses data from a specified Google Sheet.
    Uses Streamlit secrets for authentication. Caches the data for 10 minutes.

    Returns:
        pd.DataFrame: A DataFrame containing the preprocessed data, or an empty
                      DataFrame if an error occurs.
    """
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
        [span_24](start_span)service = build('sheets', 'v4', credentials=creds)[span_24](end_span)
        result = service.spreadsheets().values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        values = result.get('values', [])

        if not values:
            st.error("No data found in the Google Sheet.")
            return pd.DataFrame()

        [span_25](start_span)df = pd.DataFrame(values[1:], columns=values[0])[span_25](end_span)
        # Data Cleaning and Type Conversion
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        else:
            [span_26](start_span)st.warning("Column 'Date' not found. Time filtering will not work correctly.")[span_26](end_span)
        if 'Product' in df.columns:
            [span_27](start_span)df['Product'] = df['Product'].astype(str).str.lower().str.replace(" ", "_")[span_27](end_span)
        if 'Channel' in df.columns:
            [span_28](start_span)df['Channel'] = df['Channel'].astype(str).str.lower().str.replace(" ", "_")[span_28](end_span)
        if 'Sentimen' in df.columns:
            [span_29](start_span)df['Sentimen'] = df['Sentimen'].astype(str).str.capitalize()[span_29](end_span)
        if 'Intent' in df.columns:
            df['Intent'] = df['Intent'].astype(str)
        return df

    except KeyError as e:
        st.error(f"Missing secret: {e}. Please ensure 'gcp_service_account_credentials' is set in your Streamlit secrets.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred while loading data from Google Sheets: {e}")
        return pd.DataFrame()

@st.cache_data
def generate_health_score_data():
    """Generates static sample data for the Customer Health Score."""
    return {
        "today": {"labels": ["9 AM", "11 AM", "1 PM", "3 PM", "5 PM", "7 PM", "9 PM"], "values": [78, 76, 80, 79, 81, 83, 84], "score": 84, "trend": "+2.5%", "trend_positive": True, "trend_label": "vs. yesterday"},
        "week": {"labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "values": [79, 78, 80, 81, 83, 84, 85], "score": 85, "trend": "+1.8%", "trend_positive": True, "trend_label": "vs. last week"},
        "month": {"labels": ["Week 1", "Week 2", "Week 3", "Week 4"], "values": [79, 80, 81, 82], "score": 82, "trend": "+1.5%", "trend_positive": True, "trend_label": "vs. last month"},
        "quarter": {"labels": ["Jan", "Feb", "Mar"], "values": [76, 79, 83], "score": 83, "trend": "+3.2%", "trend_positive": True, "trend_label": "vs. last quarter"},
        "year": {"labels": ["Q1", "Q2", "Q3", "Q4"], "values": [75, 77, 80, 84], "score": 84, "trend": "+4.1%", "trend_positive": True, "trend_label": "vs. last year"},
        "all": {"labels": ["2019", "2020", "2021", "2022", "2023", "2024"], "values": [73, 71, 75, 78, 80, 83], "score": 83, "trend": "+10.4%", "trend_positive": True, "trend_label": "over 5 years"},
    }


# --- AI ASSISTANT SETUP ---

def initialize_llm_client():
    """Initializes and returns the OpenAI client for NVIDIA API."""
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )

def generate_llm_response(client, user_prompt: str, dashboard_state: dict):
    """
    Generates a streamed response from the LLM based on the user prompt and dashboard state.

    Args:
        client: The initialized OpenAI client.
        user_prompt (str): The user's question.
        dashboard_state (dict): A dictionary summarizing the current dashboard view.

    Yields:
        str: Chunks of the response from the LLM.
    """
    dashboard_summary = f"""
    Ringkasan tampilan dasbor saat ini:
    - Periode: {dashboard_state.get('time_period_label_llm', 'N/A')}
    - Skor Kesehatan: {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')})
    - Total Interaksi: {dashboard_state.get('total_interactions', 'N/A')}
    - Distribusi Sentimen: {'; [span_30](start_span)'.join([f'{k}: {v}' for k, v in dashboard_state.get('sentiment_summary', {}).items()]) if dashboard_state.get('sentiment_summary') else 'Tidak ada data.'}[span_30](end_span)
    - Distribusi Niat: {'; [span_31](start_span)'.join([f'{k}: {v}' for k, v in dashboard_state.get('intent_summary', {}).items()]) if dashboard_state.get('intent_summary') else 'Tidak ada data.'}[span_31](end_span)
    - [span_32](start_span)Ringkasan Volume: {dashboard_state.get('volume_summary', 'N/A')}[span_32](end_span)
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_VIRA},
        {"role": "user", "content": f"{dashboard_summary}\n\nPertanyaan Pengguna: \"{user_prompt}\""}
    ]

    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=messages,
            temperature=0.5,
            [span_33](start_span)top_p=0.7,[span_33](end_span)
            max_tokens=1024,
            stream=True
        )
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    except Exception as e:
        [span_34](start_span)error_message = f"Maaf, terjadi kesalahan saat menghubungi layanan AI: {str(e)}.[span_34](end_span) Silakan coba lagi nanti."
        st.error(error_message)
        yield ""

# --- CHARTING FUNCTIONS ---

def create_health_score_chart(data):
    """Creates the Plotly figure for the health score chart."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["labels"], y=data["values"], mode='lines', fill='tozeroy',
        fillcolor='rgba(52,199,89,0.18)', line=dict(color='#34c759', width=2.5)
    ))
    fig.update_layout(
        height=150, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=10)),
        yaxis=dict(showgrid=True, gridcolor='#e5e5ea', showticklabels=True, tickfont=dict(color='#4a4a4f', size=10),
                   range=[min(data["values"]) - 2, max(data["values"]) + 2])
    )
    return fig

def create_sentiment_pie_chart(data):
    """Creates the Plotly figure for the sentiment distribution pie chart."""
    if not data.empty and data['Category'].iloc[0] != 'No Data':
        fig = px.pie(data, values='Value', names='Category',
                     color='Category',
                     color_discrete_map={'Positif': '#34c759', 'Netral': '#a2a2a7', 'Negatif': '#ff3b30', 'Unknown': '#cccccc'},
                     [span_35](start_span)hole=0.7)[span_35](end_span)
        fig.update_layout(
            height=230, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=11)),
            showlegend=True)
        fig.update_traces(textinfo='percent', textfont_size=12, insidetextorientation='radial')
    else:
        fig = go.Figure(go.Indicator(mode="number", value=0, title={"text": "No Sentiment Data"}))
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_intent_bar_chart(data):
    """Creates the Plotly figure for the intent distribution bar chart."""
    if not data.empty and data['Intent'].iloc[0] != 'No Data':
        intent_color_map = {'Informasi': '#007aff', 'Keluhan': '#ff9500', 'Permohonan': '#5856d6', 'Layanan umum': '#ffcc00', 'Penutupan': '#ff3b30'}
        fig = px.bar(data, y='Intent', x='Value', orientation='h', color='Intent', color_discrete_map=intent_color_map)
        fig.update_layout(
            height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)', showlegend=False,
            xaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea'),
            yaxis=dict(title=None, categoryorder='total ascending'))
        fig.update_traces(width=0.6)
    else:
        fig = go.Figure(go.Indicator(mode="number", value=0, title={"text": "No Intent Data"}))
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_volume_line_chart(data, period_label):
    """Creates the Plotly figure for the volume trend line chart."""
    if not data.empty and data['Volume'].sum() > 0:
        fig = px.line(data, x='Day', y='Volume', line_shape='spline', markers=True)
        fig.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines+markers')
        y_min, y_max = data['Volume'].min(), data['Volume'].max()
        padding = (y_max - y_min) * 0.1
        y_range = [max(0, y_min - padding), y_max + padding + 1]
        fig.update_layout(
            height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(title=None, showgrid=False),
            yaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', range=y_range)
        )
    else:
        fig = go.Figure(go.Indicator(mode="number", value=0, title={"text": "No Volume Data"}))
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# --- UI HELPER FUNCTIONS ---

def render_sidebar():
    """Renders the sidebar navigation and information."""
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
    return page

def render_metric_card_header(title, view_options, view_key):
    """Renders the header for a metric card with a title and radio button view selector."""
    st.markdown(f'<p class="metric-title">{title}</p>', unsafe_allow_html=True)
    if view_options:
        st.radio("View", view_options, horizontal=True, key=view_key, label_visibility="collapsed")

def render_metric_card_footer():
    """Renders the closing div for a metric card."""
    st.markdown('</div>', unsafe_allow_html=True)


# --- MAIN APPLICATION ---

def main():
    """The main function to run the Streamlit application."""
    page = render_sidebar()
    master_df = load_data_from_google_sheets()

    if page == "Dashboard":
        st.title("Customer Experience Health")
        st.markdown("Real-time Insights & Performance Overview")
        st.markdown("---")

        # --- FILTERS ---
        filter_cols = st.columns(3)
        with filter_cols[0]:
            time_period_option = st.selectbox(
                "TIME PERIOD",
                ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
                index=3,
                key="time_filter"
            )
        
        available_products = sorted(list(master_df['Product'].str.replace("_", " ").str.title().unique())) if not master_df.empty and 'Product' in master_df.columns else ["N/A"]
        with filter_cols[1]:
            [span_36](start_span)selected_products = st.multiselect("PRODUCT", ["All Products"] + available_products, default=["All Products"], key="product_filter")[span_36](end_span)

        available_channels = sorted(list(master_df['Channel'].str.replace("_", " ").str.title().unique())) if not master_df.empty and 'Channel' in master_df.columns else ["N/A"]
        with filter_cols[2]:
            [span_37](start_span)selected_channels = st.multiselect("CHANNEL", ["All Channels"] + available_channels, default=["All Channels"], key="channel_filter")[span_37](end_span)

        # --- FILTERING LOGIC ---
        filtered_df = master_df.copy()
        if not filtered_df.empty and 'Date' in filtered_df.columns:
            today = pd.Timestamp('today').normalize()
            if time_period_option == "Today":
                filtered_df = filtered_df[filtered_df['Date'] == today]
            elif time_period_option == "This Week":
                [span_38](start_span)start_of_week = today - pd.to_timedelta(today.dayofweek, unit='D')[span_38](end_span)
                end_of_week = start_of_week + pd.to_timedelta(6, unit='D')
                filtered_df = filtered_df[(filtered_df['Date'] >= start_of_week) & (filtered_df['Date'] <= end_of_week)]
            elif time_period_option == "This Month":
                start_of_month = today.replace(day=1)
                end_of_month = start_of_month + pd.DateOffset(months=1) - pd.DateOffset(days=1)
                [span_39](start_span)filtered_df = filtered_df[(filtered_df['Date'] >= start_of_month) & (filtered_df['Date'] <= end_of_month)][span_39](end_span)
            elif time_period_option == "This Quarter":
                start_of_quarter = today.to_period('Q').start_time
                end_of_quarter = today.to_period('Q').end_time
                filtered_df = filtered_df[(filtered_df['Date'] >= start_of_quarter) & (filtered_df['Date'] <= end_of_quarter)]
            elif time_period_option == "This Year":
                [span_40](start_span)start_of_year = today.replace(month=1, day=1)[span_40](end_span)
                end_of_year = today.replace(month=12, day=31)
                filtered_df = filtered_df[(filtered_df['Date'] >= start_of_year) & (filtered_df['Date'] <= end_of_year)]
        
        if "All Products" not in selected_products and selected_products:
            selected_products_internal = [p.lower().replace(" ", "_") for p in selected_products]
            filtered_df = filtered_df[filtered_df['Product'].isin(selected_products_internal)]
        
        if "All Channels" not in selected_channels and selected_channels:
            [span_41](start_span)selected_channels_internal = [c.lower().replace(" ", "_") for c in selected_channels][span_41](end_span)
            filtered_df = filtered_df[filtered_df['Channel'].isin(selected_channels_internal)]

        # --- DATA PREPARATION FOR CHARTS & LLM ---
        # Health Score Data
        health_score_data_source = generate_health_score_data()
        time_period_map = {"All Periods": "all", "Today": "today", "This Week": "week", "This Month": "month", "This Quarter": "quarter", "This Year": "year"}
        current_health_data = health_score_data_source.get(time_period_map.get(time_period_option, "month"))

        # Sentiment Data
        sentiment_summary = {}
        if not filtered_df.empty and 'Sentimen' in filtered_df.columns:
            sentiment_counts = filtered_df['Sentimen'].value_counts()
            sentiment_data = sentiment_counts.reset_index()
            sentiment_data.columns = ['Category', 'Value']
            total_sentiment = sentiment_counts.sum()
            if total_sentiment > 0:
                sentiment_summary = {k: f"{(v/total_sentiment*100):.1f}% ({v} mentions)" for k, v in sentiment_counts.items()}
        else:
            sentiment_data = pd.DataFrame({'Category': ['No Data'], 'Value': [1]})

        # Intent Data
        intent_summary = {}
        if not filtered_df.empty and 'Intent' in filtered_df.columns:
            intent_counts = filtered_df['Intent'].value_counts().nlargest(5)
            intent_data = intent_counts.reset_index()
            [span_42](start_span)intent_data.columns = ['Intent', 'Value'][span_42](end_span)
            total_intent = intent_counts.sum()
            if total_intent > 0:
                intent_summary = {k: f"{(v/total_intent*100):.1f}% ({v} mentions)" for k, v in intent_counts.items()}
        else:
            intent_data = pd.DataFrame({'Intent': ['No Data'], 'Value': [1]})

        # Volume Data
        volume_summary = "Date column missing or no data."
        [span_43](start_span)if not filtered_df.empty and 'Date' in filtered_df.columns:[span_43](end_span)
            volume_over_time = filtered_df.groupby(filtered_df['Date'].dt.date)['Date'].count()
            volume_data = volume_over_time.reset_index(name='Volume')
            volume_data.columns = ['Day', 'Volume']
            if not volume_data.empty:
                [span_44](start_span)volume_summary = f"Volume trend over period: Min daily {volume_data['Volume'].min()}, Max daily {volume_data['Volume'].max()}, Avg daily {volume_data['Volume'].mean():.1f}. Total {volume_data['Volume'].sum()} interactions."[span_44](end_span)
        else:
            [span_45](start_span)volume_data = pd.DataFrame({'Day': [pd.Timestamp('today').date()], 'Volume': [0]})[span_45](end_span)

        st.markdown("---")
        # --- TOP ROW WIDGETS ---
        top_row = st.columns(3)
        with top_row[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card_header("Customer Health Score", ["Real-time", "Trend"], "health_view")
            score_cols = st.columns([1, 2])
            with score_cols[0]:
                st.markdown(f'<div class="metric-value">{current_health_data["score"]}%</div>', unsafe_allow_html=True)
            with score_cols[1]:
                trend_icon = "↑" if current_health_data["trend_positive"] else "↓"
                trend_class = "metric-trend-positive" if current_health_data["trend_positive"] else "metric-trend-negative"
                st.markdown(f'<div class="{trend_class}">{trend_icon} {current_health_data["trend"]} {current_health_data["trend_label"]}</div>', unsafe_allow_html=True)
            st.plotly_chart(create_health_score_chart(current_health_data), use_container_width=True, config={'displayModeBar': False})
            render_metric_card_footer()

        with top_row[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card_header("Critical Alerts", ["Critical", "High", "All"], "alert_view")
            st.markdown("""
            **Sudden Spike in Negative Sentiment** - *Mobile App Update X.Y: 45% negative* - *Volume: 150 mentions / 3 hrs*
            ---
            **[span_46](start_span)High Churn Risk Pattern Detected** - *Pattern: Repeated Billing Errors - Savings*[span_46](end_span)
            - *12 unique customer patterns*
            """)
            st.button("View All Alerts", type="primary", key="view_alerts")
            render_metric_card_footer()

        with top_row[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card_header("Predictive Hotspots", ["Emerging", "Trending"], "hotspot_view")
            st.markdown("""
            **[span_47](start_span)New Overdraft Policy Confusion** - *'Confused' Language: +30% WoW*[span_47](end_span)
            - *Keywords: "don't understand", "how it works"*
            ---
            **Intl. [span_48](start_span)Transfer UI Issues** - *Task Abandonment: +15% MoM*[span_48](end_span)
            - *Negative sentiment: 'Beneficiary Setup'*
            """)
            st.button("Investigate Hotspots", key="investigate_hotspots")
            render_metric_card_footer()

        # --- CUSTOMER VOICE SNAPSHOT ---
        st.markdown("## Customer Voice Snapshot")
        snapshot_cols = st.columns(3)
        with snapshot_cols[0]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card_header("Sentiment Distribution", None, None)
            st.plotly_chart(create_sentiment_pie_chart(sentiment_data), use_container_width=True, config={'displayModeBar': False})
            render_metric_card_footer()

        with snapshot_cols[1]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card_header("Top 5 Intent Distribution", None, None)
            st.plotly_chart(create_intent_bar_chart(intent_data), use_container_width=True, config={'displayModeBar': False})
            render_metric_card_footer()

        with snapshot_cols[2]:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            render_metric_card_header(f"Volume Trend ({time_period_option})", None, None)
            st.plotly_chart(create_volume_line_chart(volume_data, time_period_option), use_container_width=True, config={'displayModeBar': False})
            render_metric_card_footer()
            
        # --- VIRA CHAT ASSISTANT ---
        with st.expander("💬 Chat with VIRA - Your AI Analyst", expanded=True):
            if "messages" not in st.session_state:
                st.session_state.messages = [{"role": "assistant", "content": "Halo! Saya VIRA, asisten AI Anda. Ada yang bisa saya bantu terkait data di dasbor ini?"}]

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt := st.chat_input("Ask about insights, alerts, or trends..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    dashboard_state_for_llm = {
                        **current_health_data,
                        "time_period_label_llm": time_period_option,
                        "total_interactions": len(filtered_df),
                        "sentiment_summary": sentiment_summary,
                        "intent_summary": intent_summary,
                        "volume_summary": volume_summary,
                    }
                    
                    llm_client = initialize_llm_client()
                    stream = generate_llm_response(llm_client, prompt, dashboard_state_for_llm)
                    for chunk in stream:
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})

    else:
        st.markdown(f"## {page}")
        st.write("This section is currently under development. Please select 'Dashboard' from the sidebar.")

if __name__ == "__main__":
    main()

