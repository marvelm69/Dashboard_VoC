import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import json
from openai import OpenAI
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Google Sheets Integration
from google.oauth2 import service_account
from googleapiclient.discovery import build

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================

# Page Configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊"
)

# Google Sheets Configuration
SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']
SPREADSHEET_ID = '1V5cRgnnN5GTFsD9bR05hLzsKRWkhdEy3LhuTvSnUyIM'
RANGE_NAME = 'sheet1!A:H'

# NVIDIA API Configuration
NVIDIA_API_KEY = "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"

# ==============================================================================
# STYLING
# ==============================================================================

def apply_custom_css():
    """Apply custom CSS for a professional, simple, and appealing UI/UX"""
    st.markdown("""
    <style>
        /* Main App Styling */
        .stApp {
            background-color: #f0f2f5; /* Light grey background */
            color: #333; /* Default text color */
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; /* Softer font */
        }

        /* Sidebar Styling */
        .css-1d391kg { /* This class might change with Streamlit versions, be careful */
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }
        .css-1d391kg .stMarkdown h1 { /* VOCAL title in sidebar */
            color: #00529B; /* BCA Blue */
            text-align: center;
            font-weight: 700;
        }
        .css-1d391kg .stMarkdown h3 { /* Navigation, User Info in sidebar */
            color: #00529B;
            font-size: 1.1rem;
            font-weight: 600;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 0.3rem;
            margin-top: 1rem;
        }

        /* Main Content Area - The overall white box */
        .main > div:first-child > div:first-child { /* Target the primary content container */
            padding: 1.5rem;
            background-color: #ffffff;
            border-radius: 12px;
            margin: 1rem; /* Margin around the main white content block */
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }

        /* Content Block / Card Styling - General purpose card */
        .content-block {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e6e6e6; /* Slightly softer border */
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
            margin-bottom: 1.5rem; /* Space below each block */
            transition: box-shadow 0.3s ease;
        }
        .content-block:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .content-block h3 { /* Titles inside content blocks */
            color: #00529B; /* BCA Blue for titles */
            font-size: 1.2rem;
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 1rem;
            border-bottom: 1px solid #f0f0f0;
            padding-bottom: 0.5rem;
        }


        /* Metric Cards - Specialized for top widgets */
        .metric-card {
            background-color: #ffffff !important;
            padding: 1.5rem !important;
            border: 1px solid #e0e0e0 !important;
            box-shadow: 0 1px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            transition: box-shadow 0.3s ease;
            height: 100%; /* Make cards in a row equal height */
        }
        .metric-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .metric-card h3 { /* Titles inside metric cards */
            color: #00529B; /* BCA Blue */
            font-size: 1.15rem; /* Slightly smaller than section headers */
            font-weight: 600;
            margin-top: 0;
            margin-bottom: 0.75rem;
        }


        /* Headers and Titles */
        .dashboard-header {
            text-align: center;
            color: #003366; /* Darker BCA Blue */
            font-size: 2.4rem;
            font-weight: 700; /* Bolder */
            margin-bottom: 0.5rem; /* Reduced bottom margin */
        }
        .dashboard-subheader { /* New class for the sub-header */
            text-align: center;
            color: #555;
            font-size: 1rem;
            margin-bottom: 2rem; /* Space before filters */
            font-style: italic;
        }

        .section-header {
            font-size: 1.5rem; /* Larger section headers */
            font-weight: 600;
            color: #003366; /* Darker BCA Blue */
            border-bottom: 2px solid #007bff; /* Accent blue line */
            padding-bottom: 0.5rem;
            margin: 3rem 0 1.5rem 0; /* More space above */
        }

        /* Buttons */
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 6px; /* Slightly more rounded */
            border: none;
            padding: 10px 20px; /* More padding */
            font-weight: 500;
            transition: background-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stButton > button:hover {
            background-color: #0056b3;
            box-shadow: 0 2px 5px rgba(0, 91, 179, 0.25);
        }
        .stButton > button.secondary { /* For secondary buttons */
            background-color: #6c757d;
        }
        .stButton > button.secondary:hover {
            background-color: #5a6268;
        }


        /* Metric Values Styling */
        .metric-value {
            font-size: 2.8rem; /* Larger */
            font-weight: 700; /* Bolder */
            color: #00529B; /* BCA Blue */
            line-height: 1.1;
            text-align: center; /* Center in its container */
        }
        .metric-trend-container { /* Container for trend text */
            text-align: center; /* Center trend text */
            margin-top: 0.25rem;
        }
        .metric-trend-positive {
            color: #28a745;
            font-size: 0.95rem;
            font-weight: 600;
        }
        .metric-trend-negative {
            color: #dc3545;
            font-size: 0.95rem;
            font-weight: 600;
        }

        /* Filters */
        .filter-container { /* Already good, minor tweaks */
            background-color: #f8f9fa;
            padding: 1.2rem 1.5rem; /* Slightly more padding */
            border-radius: 8px;
            margin-bottom: 2.5rem; /* More space below filters */
            border-left: 4px solid #007bff; /* Thicker accent border */
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .filter-container .stMarkdown > h3 { /* "Filters" title inside the container */
            color: #00529B;
            font-size: 1.2rem;
            margin-top:0;
            margin-bottom: 1rem;
        }


        /* Chat Messages (keeping your good styling) */
        .stChatMessage[data-testid="stChatMessageContent"] {
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) .stChatMessage[data-testid="stChatMessageContent"] {
            background-color: #e7f3ff; /* Light blue for user */
        }
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) .stChatMessage[data-testid="stChatMessageContent"] {
            background-color: #f1f3f5; /* Light grey for assistant */
        }

        /* Alerts Styling in Widgets */
        .alert-item {
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            border-radius: 6px;
            border-left-width: 4px;
            border-left-style: solid;
        }
        .alert-critical-item { /* For items within the Alerts widget */
            background-color: #f8d7da;
            color: #721c24;
            border-left-color: #dc3545; /* Red */
        }
        .alert-warning-item { /* For items within the Alerts widget */
            background-color: #fff3cd;
            color: #856404;
            border-left-color: #ffc107; /* Yellow */
        }
        .alert-item strong {
            font-weight: 600;
        }
        .alert-item ul {
            padding-left: 20px;
            margin-top: 0.5rem;
            margin-bottom: 0.2rem;
        }

        /* Theme/Opportunity Item Styling */
        .theme-item, .opportunity-item {
            padding: 1rem;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            margin-bottom: 1rem;
            background-color: #fdfdfd;
        }
        .theme-item strong, .opportunity-item strong {
            color: #00529B; /* BCA Blue for emphasis */
            font-size: 1.05rem;
        }
        .theme-item .stMarkdown p, .opportunity-item .stMarkdown p {
            margin-bottom: 0.3rem; /* Tighter spacing for list-like items */
        }
        .theme-quote {
            font-style: italic;
            padding: 0.75rem;
            border-radius: 6px;
            margin-top: 0.5rem;
        }
        .theme-quote.positive {
            background-color: #e6ffed;
            border-left: 3px solid #28a745;
        }
        .theme-quote.negative {
            background-color: #ffebee;
            border-left: 3px solid #dc3545;
        }

        /* Helper class for vertical spacing */
        .spacer-sm { margin-bottom: 0.5rem; }
        .spacer-md { margin-bottom: 1rem; }
        .spacer-lg { margin-bottom: 2rem; }

    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# DATA LOADING & PROCESSING
# ==============================================================================

@st.cache_data(ttl=600)
def load_data_from_google_sheets():
    """Load and process data from Google Sheets with improved error handling"""
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

        result = service.spreadsheets().values().get(
            spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        
        values = result.get('values', [])
        
        if not values:
            st.error("📊 No data found in the Google Sheet.")
            return pd.DataFrame()
        
        df = pd.DataFrame(values[1:], columns=values[0])
        
        # Data preprocessing
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
            df.dropna(subset=['Date'], inplace=True)
        
        # Standardize text columns
        text_columns = ['Product', 'Channel', 'Sentimen', 'Intent']
        for col in text_columns:
            if col in df.columns:
                if col in ['Product', 'Channel']:
                    df[col] = df[col].astype(str).str.lower().str.replace(" ", "_")
                elif col == 'Sentimen':
                    df[col] = df[col].astype(str).str.capitalize()
                else:
                    df[col] = df[col].astype(str)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return pd.DataFrame()

def generate_health_score_data():
    """Generate health score data for different time periods"""
    return {
        "today": {
            "labels": ["9 AM", "11 AM", "1 PM", "3 PM", "5 PM", "7 PM", "9 PM"],
            "values": [78, 76, 80, 79, 81, 83, 84],
            "score": 84,
            "trend": "+2.5%",
            "trend_positive": True,
            "trend_label": "vs. yesterday"
        },
        "week": {
            "labels": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
            "values": [79, 78, 80, 81, 83, 84, 85],
            "score": 85,
            "trend": "+1.8%",
            "trend_positive": True,
            "trend_label": "vs. last week"
        },
        "month": {
            "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
            "values": [79, 80, 81, 82],
            "score": 82,
            "trend": "+1.5%",
            "trend_positive": True,
            "trend_label": "vs. last month"
        },
        "quarter": {
            "labels": ["Jan", "Feb", "Mar"],
            "values": [76, 79, 83],
            "score": 83,
            "trend": "+3.2%",
            "trend_positive": True,
            "trend_label": "vs. last quarter"
        },
        "year": {
            "labels": ["Q1", "Q2", "Q3", "Q4"],
            "values": [75, 77, 80, 84],
            "score": 84,
            "trend": "+4.1%",
            "trend_positive": True,
            "trend_label": "vs. last year"
        },
        "all": {
            "labels": ["2019", "2020", "2021", "2022", "2023", "2024"],
            "values": [73, 71, 75, 78, 80, 83],
            "score": 83,
            "trend": "+10.4%",
            "trend_positive": True,
            "trend_label": "over 5 years"
        }
    }

# ==============================================================================
# FILTERING FUNCTIONS
# ==============================================================================

def apply_time_filter(df, time_period):
    """Apply time-based filtering to the dataframe"""
    if df.empty or 'Date' not in df.columns:
        return df
    
    today = pd.Timestamp('today').normalize()
    
    if time_period == "Today":
        return df[df['Date'] == today]
    elif time_period == "This Week":
        start_of_week = today - pd.to_timedelta(today.dayofweek, unit='D')
        end_of_week = start_of_week + pd.to_timedelta(6, unit='D')
        return df[(df['Date'] >= start_of_week) & (df['Date'] <= end_of_week)]
    elif time_period == "This Month":
        start_of_month = today.replace(day=1)
        end_of_month = (start_of_month + pd.DateOffset(months=1)) - pd.DateOffset(days=1)
        return df[(df['Date'] >= start_of_month) & (df['Date'] <= end_of_month)]
    elif time_period == "This Quarter":
        start_of_quarter = today.to_period('Q').start_time
        end_of_quarter = today.to_period('Q').end_time
        return df[(df['Date'] >= start_of_quarter) & (df['Date'] <= end_of_quarter)]
    elif time_period == "This Year":
        start_of_year = today.replace(month=1, day=1)
        end_of_year = today.replace(month=12, day=31)
        return df[(df['Date'] >= start_of_year) & (df['Date'] <= end_of_year)]
    
    return df

def apply_product_filter(df, selected_products):
    """Apply product filtering to the dataframe"""
    if "All Products" not in selected_products and selected_products and 'Product' in df.columns:
        selected_products_internal = [p.lower().replace(" ", "_") for p in selected_products]
        return df[df['Product'].isin(selected_products_internal)]
    return df

def apply_channel_filter(df, selected_channels):
    """Apply channel filtering to the dataframe"""
    if "All Channels" not in selected_channels and selected_channels and 'Channel' in df.columns:
        selected_channels_internal = [c.lower().replace(" ", "_") for c in selected_channels]
        return df[df['Channel'].isin(selected_channels_internal)]
    return df

# ==============================================================================
# VISUALIZATION FUNCTIONS
# ==============================================================================

def create_health_score_chart(data):
    """Create an enhanced health score chart"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=data["labels"],
        y=data["values"],
        mode='lines+markers',
        fill='tozeroy',
        fillcolor='rgba(52,199,89,0.2)',
        line=dict(color='#34c759', width=3),
        marker=dict(size=8, color='#34c759'),
        name='Health Score'
    ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickfont=dict(color='#4a4a4f', size=10)
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(229,229,234,0.5)',
            showline=False,
            showticklabels=True,
            tickfont=dict(color='#4a4a4f', size=10),
            range=[min(data["values"]) - 2, max(data["values"]) + 2]
        )
    )
    
    return fig

def create_sentiment_chart(sentiment_data):
    """Create an enhanced sentiment distribution chart"""
    if sentiment_data.empty or sentiment_data['Category'].iloc[0] == 'No Data':
        fig = go.Figure(go.Indicator(
            mode="number",
            value=0,
            title={"text": "No Sentiment Data Available"}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    colors = {'Positif': '#34c759', 'Netral': '#a2a2a7', 'Negatif': '#ff3b30', 'Unknown': '#cccccc'}
    
    fig = px.pie(
        sentiment_data,
        values='Value',
        names='Category',
        color='Category',
        color_discrete_map=colors,
        hole=0.6
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.3,
            xanchor='center',
            x=0.5,
            font=dict(size=11)
        ),
        showlegend=True
    )
    
    fig.update_traces(
        textinfo='percent+label',
        textfont_size=11,
        insidetextorientation='radial'
    )
    
    return fig

def create_intent_chart(intent_data):
    """Create an enhanced intent distribution chart"""
    if intent_data.empty or intent_data['Intent'].iloc[0] == 'No Data':
        fig = go.Figure(go.Indicator(
            mode="number",
            value=0,
            title={"text": "No Intent Data Available"}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    intent_colors = ['#007aff', '#ff9500', '#5856d6', '#ffcc00', '#ff3b30']
    
    fig = px.bar(
        intent_data,
        y='Intent',
        x='Value',
        orientation='h',
        color='Intent',
        color_discrete_sequence=intent_colors
    )
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=10, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='rgba(229,229,234,0.5)',
            showline=False,
            showticklabels=True
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            showline=False,
            showticklabels=True,
            categoryorder='total ascending'
        ),
        showlegend=False
    )
    
    fig.update_traces(marker_line_width=0, width=0.7)
    
    return fig

def create_volume_chart(volume_data):
    """Create an enhanced volume trend chart"""
    if volume_data.empty or volume_data['Volume'].sum() == 0:
        fig = go.Figure(go.Indicator(
            mode="number",
            value=0,
            title={"text": "No Volume Data Available"}
        ))
        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
        return fig
    
    fig = px.line(
        volume_data,
        x='Day',
        y='Volume',
        line_shape='spline',
        markers=True
    )
    
    fig.update_traces(
        line_color='#007aff',
        fill='tozeroy',
        fillcolor='rgba(0,122,255,0.2)',
        mode='lines+markers',
        marker=dict(size=6, color='#007aff')
    )
    
    y_min = volume_data['Volume'].min()
    y_max = volume_data['Volume'].max()
    y_range = [max(0, y_min - (y_max - y_min) * 0.1), y_max + (y_max - y_min) * 0.1 + 1]
    
    fig.update_layout(
        height=250,
        margin=dict(l=0, r=10, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=None,
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='rgba(229,229,234,0.5)',
            showline=False,
            showticklabels=True,
            tickfont=dict(size=10),
            range=y_range
        )
    )
    
    return fig

# ==============================================================================
# ANALYTICS FUNCTIONS
# ==============================================================================

def process_filtered_data(df):
    """Process filtered data and return analytics summaries"""
    total_interactions = len(df)
    
    # Sentiment Analysis
    if not df.empty and 'Sentimen' in df.columns:
        sentiment_counts = df['Sentimen'].value_counts()
        sentiment_data = sentiment_counts.reset_index()
        sentiment_data.columns = ['Category', 'Value']
        
        total_sentiment = sentiment_counts.sum()
        sentiment_summary = {
            k: f"{(v/total_sentiment*100):.1f}% ({v} mentions)"
            for k, v in sentiment_counts.items()
        } if total_sentiment > 0 else {}
    else:
        sentiment_data = pd.DataFrame({'Category': ['No Data'], 'Value': [1]})
        sentiment_summary = {"Info": "No sentiment data for current filter."}
    
    # Intent Analysis
    if not df.empty and 'Intent' in df.columns:
        intent_counts = df['Intent'].value_counts().nlargest(5)
        intent_data = intent_counts.reset_index()
        intent_data.columns = ['Intent', 'Value']
        
        total_intent = intent_counts.sum()
        intent_summary = {
            k: f"{(v/total_intent*100):.1f}% ({v} mentions)"
            for k, v in intent_counts.items()
        } if total_intent > 0 else {}
    else:
        intent_data = pd.DataFrame({'Intent': ['No Data'], 'Value': [1]})
        intent_summary = {"Info": "No intent data for current filter."}
    
    # Volume Analysis
    if not df.empty and 'Date' in df.columns:
        volume_over_time = df.groupby(df['Date'].dt.date)['Date'].count()
        volume_data = volume_over_time.reset_index(name='Volume')
        volume_data.columns = ['Day', 'Volume']
        
        if not volume_data.empty:
            volume_summary = (
                f"Volume trend: Min daily {volume_data['Volume'].min()}, "
                f"Max daily {volume_data['Volume'].max()}, "
                f"Avg daily {volume_data['Volume'].mean():.1f}. "
                f"Total {volume_data['Volume'].sum()} interactions."
            )
        else:
            volume_summary = "No volume data to display for the selected filters."
            volume_data = pd.DataFrame({'Day': [pd.Timestamp('today').date()], 'Volume': [0]})
    else:
        volume_data = pd.DataFrame({'Day': [pd.Timestamp('today').date()], 'Volume': [0]})
        volume_summary = "Volume data cannot be trended (Date column missing or no data)."
    
    return {
        'total_interactions': total_interactions,
        'sentiment_data': sentiment_data,
        'sentiment_summary': sentiment_summary,
        'intent_data': intent_data,
        'intent_summary': intent_summary,
        'volume_data': volume_data,
        'volume_summary': volume_summary
    }

# ==============================================================================
# AI CHAT FUNCTIONS
# ==============================================================================

SYSTEM_PROMPT_VIRA = """
Anda adalah VIRA, seorang konsultan virtual untuk Bank BCA.
Tugas utama Anda adalah menganalisis data dasbor yang disediakan dan memberikan wawasan, ringkasan, serta saran yang relevan.
Fokuslah pada metrik seperti skor kesehatan, tren, sentimen pelanggan, niat panggilan, dan volume panggilan berdasarkan data yang disaring.
Selalu dasarkan jawaban Anda pada data yang diberikan dalam `dashboard_state`.
Gunakan bahasa Indonesia yang sopan dan mudah dimengerti.
Jika ada pertanyaan yang tidak dapat dijawab dari data dasbor, sampaikan dengan sopan bahwa informasi tersebut tidak tersedia.
Berikan analisis yang ringkas namun mendalam.
Jika ada pertanyaan di luar konteks analisis Anda, sampaikan bahwa itu di luar kapabilitas Anda.
"""
def get_top_themes(df, sentiment_type, top_n=3):
    """
    Mengidentifikasi top N 'Intent' berdasarkan sentimen tertentu dan menghitung metriknya.
    """
    if df.empty or 'Intent' not in df.columns or 'Sentimen' not in df.columns:
        return []

    # Filter berdasarkan sentimen
    df_sentiment_filtered = df[df['Sentimen'] == sentiment_type]

    if df_sentiment_filtered.empty:
        return []

    # Hitung jumlah kemunculan setiap Intent dan rata-rata sentimen (walaupun sudah difilter,
    # ini berguna jika ada sub-skor sentimen numerik nantinya)
    # Untuk sekarang, kita hanya hitung frekuensi Intent
    theme_counts = df_sentiment_filtered['Intent'].value_counts().nlargest(top_n)

    top_themes_data = []
    for intent, mentions in theme_counts.items():
        # Ambil contoh verbatim (Summary aktual) untuk tema ini
        # Ambil Summary pertama yang cocok dengan intent dan sentimen ini
        example_comment_series = df_sentiment_filtered[df_sentiment_filtered['Intent'] == intent]['Summary'] # Asumsi ada kolom 'Summary'
        example_comment = example_comment_series.iloc[0] if not example_comment_series.empty else f"No example comment for {intent}."

        top_themes_data.append({
            "theme": intent, # 'Intent' sekarang menjadi 'theme'
            "mentions": mentions,
            "sentiment_type": sentiment_type, # Menyimpan tipe sentimen
            "example_comment": example_comment
        })
    return top_themes_data
def initialize_ai_client():
    """Initialize the AI client"""
    return OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY
    )

def generate_llm_response(user_prompt, dashboard_state, system_prompt):
    """Generate AI response based on dashboard state"""
    client = initialize_ai_client()
    
    dashboard_summary = f"""
Ringkasan dasbor saat ini:
- Periode: {dashboard_state.get('time_period_label_llm', 'N/A')}
- Skor Kesehatan: {dashboard_state.get('score', 'N/A')}% (Tren: {dashboard_state.get('trend', 'N/A')} - {dashboard_state.get('trend_label', 'N/A')})
- Total Interaksi: {dashboard_state.get('total_interactions', 'N/A')}
- Distribusi Sentimen: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('sentiment_summary', {}).items()]) if dashboard_state.get('sentiment_summary') else 'Tidak ada data sentimen.'}
- Distribusi Niat: {'; '.join([f"{k}: {v}" for k, v in dashboard_state.get('intent_summary', {}).items()]) if dashboard_state.get('intent_summary') else 'Tidak ada data niat.'}
- Tren Volume: {dashboard_state.get('volume_summary', 'N/A')}
"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{dashboard_summary}\n\nPertanyaan: \"{user_prompt}\""}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
            messages=messages,
            temperature=0.5,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )
        
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        error_message = f"Maaf, terjadi kesalahan: {str(e)}. Silakan coba lagi."
        yield error_message

# ==============================================================================
# UI COMPONENTS
# ==============================================================================

def render_sidebar():
    """Render the sidebar navigation"""
    with st.sidebar:
        st.markdown("# 🎯 VOCAL") # This will be styled by .css-1d391kg .stMarkdown h1
        st.markdown("---")

        st.markdown("### 📊 Navigation") # Styled by .css-1d391kg .stMarkdown h3
        page = st.selectbox(
            "Choose Page",
            ["Dashboard", "Analytics", "Feedback", "Alerts", "Reports"],
            label_visibility="collapsed", # Hide the label, "Navigation" acts as it
            key="menu_nav"
        )

        st.markdown("### 👤 User Info") # Styled by .css-1d391kg .stMarkdown h3
        st.markdown("**Sebastian**")
        st.markdown("*CX Manager*")
        st.markdown("---")
        st.caption(f"Version 1.0.0 | Last updated: {datetime.now().strftime('%Y-%m-%d')}")

        return page

def render_filters(master_df):
    """Render the filter controls"""
    # Apply the filter-container class
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    st.markdown("### 🔧 Filters") # This H3 will be styled by .filter-container .stMarkdown > h3

    col1, col2, col3 = st.columns(3)

    with col1:
        time_period = st.selectbox(
            "⏰ Time Period",
            ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
            index=3, # Default to "This Month"
            key="time_filter"
        )
    # ... (rest of your filter logic is fine) ...
    if not master_df.empty and 'Product' in master_df.columns:
        # Convert product names for display: replace underscores, title case
        unique_products = master_df['Product'].unique()
        display_products = sorted([p.replace("_", " ").title() for p in unique_products])
        available_products = display_products
    else:
        available_products = ["MyBCA", "BCA Mobile", "KPR", "KKB", "KSM", "Investasi", "Asuransi"]

    if not master_df.empty and 'Channel' in master_df.columns:
        unique_channels = master_df['Channel'].unique()
        display_channels = sorted([c.replace("_", " ").title() for c in unique_channels])
        available_channels = display_channels
    else:
        available_channels = ["Social Media", "Call Center", "WhatsApp", "Webchat", "VIRA", "E-mail"]

    with col2:
        selected_products = st.multiselect(
            "🏦 Products",
            ["All Products"] + available_products,
            default=["All Products"],
            key="product_filter"
        )

    with col3:
        selected_channels = st.multiselect(
            "📱 Channels",
            ["All Channels"] + available_channels,
            default=["All Channels"],
            key="channel_filter"
        )

    st.markdown('</div>', unsafe_allow_html=True) # Close filter-container

    return time_period, selected_products, selected_channels

def render_health_score_widget(health_data):
    """Render the health score widget"""
    # INI DIA PEMBUKA KOTAKNYA
    st.markdown('<div class="metric-card">', unsafe_allow_html=True) # Open metric-card

    st.markdown("<h3>💚 Customer Health Score</h3>", unsafe_allow_html=True) # Card title

    # Display score and trend more prominently
    col_score, col_trend_text = st.columns([2,3])
    with col_score:
        st.markdown(f'<div class="metric-value">{health_data["score"]}%</div>', unsafe_allow_html=True)

    with col_trend_text:
        trend_icon = "📈" if health_data["trend_positive"] else "📉"
        trend_class = "metric-trend-positive" if health_data["trend_positive"] else "metric-trend-negative"
        st.markdown(f'''
            <div class="metric-trend-container">
                <span class="{trend_class}">{trend_icon} {health_data["trend"]}</span><br>
                <span style="font-size:0.8em; color: #666;">{health_data["trend_label"]}</span>
            </div>
        ''', unsafe_allow_html=True)

    # Health score chart
    fig_health = create_health_score_chart(health_data)
    st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})

    # Health score interpretation
    score = health_data["score"]
    if score >= 80:
        st.success("🎉 Excellent customer satisfaction! Keep up the great work.", icon="✅")
    elif score >= 70:
        st.info("👍 Good customer satisfaction with room for improvement.", icon="💡")
    elif score >= 60:
        st.warning("⚠️ Moderate satisfaction. Consider addressing key issues.", icon="🔍")
    else:
        st.error("🚨 Low satisfaction detected. Immediate action recommended.", icon="🔥")

    # DAN INI DIA PENUTUP KOTAKNYA
    st.markdown('</div>', unsafe_allow_html=True) # Close metric-card

def render_alerts_widget():
   """Render the critical alerts widget"""
   st.markdown('<div class="metric-card">', unsafe_allow_html=True) # Open metric-card
   st.markdown("<h3>🚨 Critical Alerts</h3>", unsafe_allow_html=True) # Card title

   # Critical alert item
   st.markdown("""
   <div class="alert-item alert-critical-item">
       <strong>🔴 Sudden Spike in Negative Sentiment</strong>
       <ul>
           <li>Product: Mobile App Update X.Y</li>
           <li>Details: 45% negative sentiment, 150 mentions (last 3 hrs)</li>
           <li>Key Issues: Login failures, App crashes</li>
           <li><strong>Action Required:</strong> Immediate technical review</li>
       </ul>
   </div>
   """, unsafe_allow_html=True)

   # High priority alert item
   st.markdown("""
   <div class="alert-item alert-warning-item">
       <strong>🟡 High Churn Risk Pattern</strong>
       <ul>
           <li>Pattern: Repeated billing errors in Savings accounts</li>
           <li>Affected: 12 unique customer patterns identified</li>
           <li>Avg. Sentiment: -0.8 (Very Negative)</li>
           <li><strong>Action:</strong> Customer retention outreach recommended</li>
       </ul>
   </div>
   """, unsafe_allow_html=True)

   # Buttons at the bottom of the card
   st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True) # Add a little space
   col1, col2 = st.columns(2)
   with col1:
       if st.button("🔍 View All Alerts", key="alerts_view_all", type="primary", use_container_width=True):
           st.toast("Redirecting to detailed alerts page...", icon=" M")

   with col2:
       if st.button("📋 Create Action Plan", key="alerts_action_plan", type="secondary", use_container_width=True):
           st.toast("Opening action plan creator...", icon="✍️")

   st.markdown('</div>', unsafe_allow_html=True) # Close metric-card

def render_hotspots_widget():
   """Render the predictive hotspots widget"""
   st.markdown('<div class="metric-card">', unsafe_allow_html=True) # Open metric-card
   st.markdown("<h3>🔮 Predictive Hotspots</h3>", unsafe_allow_html=True) # Card title

   # Emerging issues
   st.markdown("""
   <div class="alert-item alert-warning-item" style="border-left-color: #ff9500;"> <!-- Custom orange for this one -->
       <strong>🆕 New Overdraft Policy Confusion</strong>
       <ul>
           <li>Impact Level: Medium 🟡</li>
           <li>'Confused' language patterns: +30% WoW</li>
           <li>Common phrases: "don't understand", "how it works"</li>
           <li><strong>Recommendation:</strong> Create clearer policy explanation</li>
       </ul>
   </div>
   """, unsafe_allow_html=True)

   st.markdown("""
   <div class="alert-item" style="background-color: #e7f3ff; color: #004085; border-left-color: #007bff;"> <!-- Custom info blue -->
       <strong>🌐 International Transfer UI Issues</strong>
       <ul>
           <li>Impact Level: Low 🟢</li>
           <li>Task abandonment rate: +15% MoM</li>
           <li>Negative sentiment around 'Beneficiary Setup'</li>
           <li><strong>Recommendation:</strong> UI/UX review for international transfers</li>
       </ul>
   </div>
   """, unsafe_allow_html=True)

   # Trend indicators
   st.markdown('<div class="spacer-sm"></div>', unsafe_allow_html=True)
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("Emerging Issues", "2", delta="1")
   with col2:
       st.metric("Trending Topics", "5", delta="2")
   with col3:
       st.metric("Predicted Risks", "3", delta="-1")

   st.markdown('</div>', unsafe_allow_html=True) # Close metric-card

def render_voice_snapshot(analytics_data, time_period):
   """Render the customer voice snapshot section"""
   st.markdown('<div class="section-header">📊 Customer Voice Snapshot</div>', unsafe_allow_html=True)

   col1, col2, col3 = st.columns(3)

   with col1:
       st.markdown('<div class="content-block">', unsafe_allow_html=True) # Open content-block
       st.markdown("<h3>😊 Sentiment Distribution</h3>", unsafe_allow_html=True)

       fig_sentiment = create_sentiment_chart(analytics_data['sentiment_data'])
       st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})

       sentiment_summary = analytics_data['sentiment_summary']
       if 'Positif' in sentiment_summary:
           st.success(f"Positive: {sentiment_summary['Positif'].split('(')[0].strip()}", icon="👍")
       if 'Negatif' in sentiment_summary:
           st.error(f"Negative: {sentiment_summary['Negatif'].split('(')[0].strip()}", icon="👎")
       if 'Netral' in sentiment_summary:
           st.info(f"Neutral: {sentiment_summary['Netral'].split('(')[0].strip()}", icon="💬")

       st.markdown('</div>', unsafe_allow_html=True) # Close content-block

   with col2:
       st.markdown('<div class="content-block">', unsafe_allow_html=True) # Open content-block
       st.markdown("<h3>🎯 Intent Distribution (Top 5)</h3>", unsafe_allow_html=True)

       fig_intent = create_intent_chart(analytics_data['intent_data'])
       st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})

       intent_summary = analytics_data['intent_summary']
       if intent_summary and "Info" not in intent_summary and intent_summary.keys():
           top_intent = list(intent_summary.keys())[0]
           st.info(f"Top intent: **{top_intent}** ({intent_summary[top_intent].split('(')[0].strip()})", icon="🎯")
       else:
           st.caption("No dominant intent or data unavailable.")

       st.markdown('</div>', unsafe_allow_html=True) # Close content-block

   with col3:
       st.markdown('<div class="content-block">', unsafe_allow_html=True) # Open content-block
       st.markdown(f"<h3>📈 Volume Trend ({time_period})</h3>", unsafe_allow_html=True)

       fig_volume = create_volume_chart(analytics_data['volume_data'])
       st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})

       total_interactions = analytics_data['total_interactions']
       st.metric("Total Interactions", f"{total_interactions:,}") # Format number

       st.markdown('</div>', unsafe_allow_html=True) # Close content-block
       
def render_customer_themes(positive_themes_data, negative_themes_data):
   """Render the customer themes section based on processed data."""
   st.markdown('<div class="section-header">💭 Top Customer Themes (Based on Intent & Sentiment)</div>', unsafe_allow_html=True)

   col1, col2 = st.columns(2)

   with col1:
       st.markdown('<div class="content-block">', unsafe_allow_html=True)
       st.markdown("<h3>🌟 Top Positive Themes</h3>", unsafe_allow_html=True)

       if positive_themes_data:
           for theme_info in positive_themes_data:
               st.markdown(f"""
               <div class="theme-item">
                   <strong>{theme_info['theme']}</strong>
                   <p style="margin-bottom:0.1rem;">Mentions: {theme_info['mentions']}</p>
               </div>
               """, unsafe_allow_html=True)
           # Menampilkan contoh Summary dari tema positif pertama (jika ada)
           if positive_themes_data[0].get("example_comment"):
                st.markdown(f'<div class="theme-quote positive">💬 "{positive_themes_data[0]["example_comment"]}"</div>', unsafe_allow_html=True)
       else:
           st.info("No significant positive themes found for the current filters.")

       st.markdown('</div>', unsafe_allow_html=True)

   with col2:
       st.markdown('<div class="content-block">', unsafe_allow_html=True)
       st.markdown("<h3>⚠️ Top Negative Themes</h3>", unsafe_allow_html=True)

       if negative_themes_data:
           for theme_info in negative_themes_data:
               st.markdown(f"""
               <div class="theme-item">
                   <strong>{theme_info['theme']}</strong>
                   <p style="margin-bottom:0.1rem;">Mentions: {theme_info['mentions']}</p>
               </div>
               """, unsafe_allow_html=True)
           # Menampilkan contoh Summary dari tema negatif pertama (jika ada)
           if negative_themes_data[0].get("example_comment"):
                st.markdown(f'<div class="theme-quote negative">💬 "{negative_themes_data[0]["example_comment"]}"</div>', unsafe_allow_html=True)
       else:
           st.info("No significant negative themes found for the current filters.")
       st.markdown('</div>', unsafe_allow_html=True)

   col1, col2, col3 = st.columns(3)

   with col1:
       st.markdown('<div class="content-block">', unsafe_allow_html=True)
       st.markdown("<h3>🎉 Delightful Features</h3>", unsafe_allow_html=True)
       st.markdown("""
       <div class="opportunity-item">
           <strong>Instant Card Activation</strong>
           <p>Delight mentions: 75 this week</p>
           <p>Sentiment Score: +0.95 (Exceptional)</p>
           <p>Keywords: "amazing", "so easy", "instant"</p>
           <p><strong>Opportunity:</strong> Amplify in marketing</p>
           <p><strong>ROI Potential:</strong> <span style="color:green; font-weight:bold;">High</span></p>
       </div>
       """, unsafe_allow_html=True)
       st.success("**Action**: Showcase in customer testimonials", icon="💡")
       st.markdown('</div>', unsafe_allow_html=True)

   with col2:
       st.markdown('<div class="content-block">', unsafe_allow_html=True)
       st.markdown("<h3>💰 Cross-Sell Opportunities</h3>", unsafe_allow_html=True)
       st.markdown("""
       <div class="opportunity-item">
           <strong>Mortgage Inquiry Surge</strong>
           <p>Mortgage info requests: +15% WoW</p>
           <p>Related topics: Savings, Financial Planning</p>
           <p>Segments: 25-40 age group</p>
           <p><strong>Opportunity:</strong> Targeted mortgage promotions</p>
           <p><strong>ROI Potential:</strong> <span style="color:darkorange; font-weight:bold;">Very High</span></p>
       </div>
       """, unsafe_allow_html=True)
       st.info("**Action**: Create personalized mortgage offers", icon="📈")
       st.markdown('</div>', unsafe_allow_html=True)

   with col3:
       st.markdown('<div class="content-block">', unsafe_allow_html=True)
       st.markdown("<h3>⭐ Service Excellence</h3>", unsafe_allow_html=True)
       st.markdown("""
       <div class="opportunity-item">
           <strong>Complex Issue Resolution</strong>
           <p>Excellence mentions: 25 (complex cases)</p>
           <p>Top performers: Agent A, B, C</p>
           <p>Resolution time: 15% faster than avg</p>
           <p><strong>Opportunity:</strong> Scale best practices</p>
           <p><strong>ROI Potential:</strong> <span style="color:green; font-weight:bold;">Medium-High</span></p>
       </div>
       """, unsafe_allow_html=True)
       st.success("**Action**: Implement training program", icon="🤝")
       st.markdown('</div>', unsafe_allow_html=True)
       
def render_vira_chat(dashboard_state):
   """Render the VIRA chat interface"""
   st.markdown('<div class="section-header">🤖 Chat with VIRA (Your CX Co-pilot)</div>', unsafe_allow_html=True)
   st.markdown('<div class="content-block">', unsafe_allow_html=True) # Wrap chat in a content block
   
    # Initialize chat history
   if "messages" not in st.session_state:
       st.session_state.messages = [
           {
               "role": "assistant",
               "content": "🙋‍♀️ Halo! Saya VIRA, asisten AI Anda untuk analisis Voice of Customer. "
                         "Saya dapat membantu menganalisis data dasbor, memberikan insights, dan menjawab pertanyaan "
                         "terkait performa customer experience. Ada yang bisa saya bantu hari ini?"
           }
       ]

   # Display chat history
   for message in st.session_state.messages:
       with st.chat_message(message["role"]):
           st.markdown(message["content"])

   # Chat input
   if prompt := st.chat_input("💬 Tanyakan tentang insights, trends, atau analisis data..."):
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.markdown(prompt)

       with st.chat_message("assistant"):
           message_placeholder = st.empty()
           full_response = ""
           try:
               with st.spinner("VIRA sedang menganalisis dan berpikir..."):
                   stream = generate_llm_response(prompt, dashboard_state, SYSTEM_PROMPT_VIRA)
                   for chunk in stream:
                       full_response += chunk
                       message_placeholder.markdown(full_response + "▌")
                   message_placeholder.markdown(full_response)
           except Exception as e:
               error_message = f"🚫 Maaf, terjadi kesalahan dengan VIRA: {str(e)}. Silakan coba lagi."
               message_placeholder.error(error_message)
               full_response = error_message
       st.session_state.messages.append({"role": "assistant", "content": full_response})

   # Chat controls
   st.markdown('<div class="spacer-md"></div>', unsafe_allow_html=True) # Add space before buttons
   col1, col2, col3 = st.columns([1, 1, 1])
   with col1:
       if st.button("🔄 Clear Chat", type="secondary", use_container_width=True, key="clear_chat_vira"):
           st.session_state.messages = [
               {
                   "role": "assistant",
                   "content": "Riwayat chat telah dihapus. Ada yang bisa saya bantu?"
               }
           ]
           st.rerun()
   with col2:
       if st.button("💾 Export Chat", type="secondary", use_container_width=True, key="export_chat_vira"):
           chat_export = "\n".join([
               f"{msg['role'].title()}: {msg['content']}"
               for msg in st.session_state.messages
           ])
           st.download_button(
               "📥 Download Chat",
               chat_export,
               "vira_chat_export.txt",
               "text/plain",
               use_container_width=True
           )
   with col3:
       if st.button("❓ Chat Help", type="secondary", use_container_width=True, key="help_chat_vira"):
           st.sidebar.info("""
           **💡 Tips Menggunakan VIRA:**
           - Tanyakan ringkasan data saat ini.
           - Minta analisis tren sentimen atau volume.
           - Tanyakan tentang korelasi antar metrik.
           - Minta VIRA mengidentifikasi anomali.
           - "Berikan saya 3 rekomendasi aksi."
           """, icon="💡")
           st.toast("Bantuan VIRA ditampilkan di sidebar.", icon="ℹ️")

   st.markdown('</div>', unsafe_allow_html=True) # Close content-block
# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
   """Main application function"""
   apply_custom_css()

   with st.spinner("🔄 Loading data from Google Sheets... This might take a moment."):
       master_df = load_data_from_google_sheets()

   current_page = render_sidebar()

   if current_page == "Dashboard":
       st.markdown('<div class="dashboard-header">🏦 Voice of Customer Dashboard</div>', unsafe_allow_html=True)
       st.markdown('<div class="dashboard-subheader">*Real-time Customer Experience Insights & Performance Analytics*</div>', unsafe_allow_html=True)

       time_period, selected_products, selected_channels = render_filters(master_df)

       filtered_df = master_df.copy()
       if not filtered_df.empty:
            filtered_df = apply_time_filter(filtered_df, time_period)
            filtered_df = apply_product_filter(filtered_df, selected_products)
            filtered_df = apply_channel_filter(filtered_df, selected_channels)
       else:
            st.warning("Master data is empty. Cannot apply filters. Displaying with dummy data or empty states.")

       analytics_data = process_filtered_data(filtered_df)

       health_score_data = generate_health_score_data()
       time_period_map = {
           "All Periods": "all", "Today": "today", "This Week": "week",
           "This Month": "month", "This Quarter": "quarter", "This Year": "year"
       }
       selected_time_key = time_period_map.get(time_period, "month")
       current_health_data = health_score_data[selected_time_key].copy()
       current_health_data['time_period_label'] = time_period

       cols_kpi = st.columns(3)
       with cols_kpi[0]:
           render_health_score_widget(current_health_data)
       with cols_kpi[1]:
           render_alerts_widget()
       with cols_kpi[2]:
           render_hotspots_widget()

       render_voice_snapshot(analytics_data, time_period)

       if not filtered_df.empty and analytics_data['total_interactions'] > 0 :
           summary_parts = []
           sentiment_summary = analytics_data['sentiment_summary']
           intent_summary = analytics_data['intent_summary']

           if 'Positif' in sentiment_summary:
               positive_pct_str = sentiment_summary['Positif'].split('%')[0]
               try:
                   positive_pct = float(positive_pct_str)
                   if positive_pct > 60:
                       summary_parts.append(f"🌟 Sentimen positif dominan ({positive_pct_str}%)")
                   else:
                       summary_parts.append(f"Sentimen positif ({positive_pct_str}%)")
               except ValueError:
                    summary_parts.append(f"Sentimen positif: {positive_pct_str}%")

           if intent_summary and "Info" not in intent_summary and intent_summary.keys():
               top_intent = list(intent_summary.keys())[0]
               summary_parts.append(f"🎯 '{top_intent}' adalah niat utama")

           if analytics_data['total_interactions'] > 0:
               summary_parts.append(f"dari total {analytics_data['total_interactions']:,} interaksi")

           if summary_parts:
               st.success(f"**📈 Ringkasan Data ({time_period}):** {', '.join(summary_parts)}.", icon="💡")
       elif not master_df.empty:
           st.warning(f"⚠️ Tidak ada data yang ditemukan untuk filter yang dipilih ({time_period}, Produk: {selected_products}, Channel: {selected_channels}). Coba ubah filter Anda.", icon="🚫")

       # =======================================================================
       # >>> AWAL BAGIAN YANG DIMODIFIKASI/DITAMBAHKAN UNTUK CUSTOMER THEMES <<<
       # =======================================================================

       # Dapatkan data untuk Top Customer Themes
       # Pastikan kolom 'Summary' ada di filtered_df jika ingin example_comment berfungsi.
       # GANTI 'Summary' DENGAN NAMA KOLOM VERBATIM ANDA JIKA BERBEDA.
       # Kolom yang dibutuhkan: 'Intent', 'Sentimen', dan kolom verbatim (misal, 'Summary')
       nama_kolom_verbatim = 'Summary' # <-- GANTI INI JIKA PERLU
       required_theme_cols = ['Intent', 'Sentimen', nama_kolom_verbatim]

       top_positive_themes = []
       top_negative_themes = []

       if not filtered_df.empty: # Hanya proses jika filtered_df tidak kosong
           if all(col in filtered_df.columns for col in required_theme_cols):
               top_positive_themes = get_top_themes(filtered_df, 'Positif', top_n=3) # 'Positif' harus cocok dengan data Anda
               top_negative_themes = get_top_themes(filtered_df, 'Negatif', top_n=3) # 'Negatif' harus cocok dengan data Anda
           else:
               missing_cols = [col for col in required_theme_cols if col not in filtered_df.columns]
               st.warning(f"Kolom yang dibutuhkan untuk Customer Themes ({', '.join(missing_cols)}) tidak ditemukan di data. Themes tidak dapat ditampilkan.", icon="⚠️")
       # Jika filtered_df kosong, top_positive_themes dan top_negative_themes akan tetap list kosong.

       # Panggil render_customer_themes dengan data yang sudah diproses
       # Ini MENGGANTIKAN panggilan render_customer_themes() yang lama (jika ada yang menggunakan data dummy)
       render_customer_themes(top_positive_themes, top_negative_themes)

       # =======================================================================
       # >>> AKHIR BAGIAN YANG DIMODIFIKASI/DITAMBAHKAN UNTUK CUSTOMER THEMES <<<
       # =======================================================================

       dashboard_state = {
           **current_health_data,
           "time_period_label_llm": time_period,
           'total_interactions': analytics_data.get('total_interactions', 'N/A'),
           'sentiment_summary': analytics_data.get('sentiment_summary', {}),
           'intent_summary': analytics_data.get('intent_summary', {}),
           'volume_summary': analytics_data.get('volume_summary', 'N/A'),
           'critical_alerts_summary': "Terdapat lonjakan sentimen negatif terkait update aplikasi mobile dan pola risiko churn pada rekening tabungan.",
           'emerging_hotspots_summary': "Isu baru terkait kebingungan kebijakan overdraft dan masalah UI transfer internasional.",
           # Tambahkan ringkasan tema ke dashboard_state jika VIRA perlu mengetahuinya
           'positive_themes_summary': f"{len(top_positive_themes)} tema positif teratas: " + ", ".join([t['theme'] for t in top_positive_themes]) if top_positive_themes else "Tidak ada tema positif signifikan.",
           'negative_themes_summary': f"{len(top_negative_themes)} tema negatif teratas: " + ", ".join([t['theme'] for t in top_negative_themes]) if top_negative_themes else "Tidak ada tema negatif signifikan."
       }
       render_vira_chat(dashboard_state)

   else:
       # ... (kode untuk halaman lain) ...
       st.markdown(f'<div class="dashboard-header">🚧 {current_page} - Segera Hadir</div>', unsafe_allow_html=True)
       st.markdown(f"*Fitur canggih untuk halaman '{current_page}' sedang dalam tahap akhir pengembangan.*")
       st.info(f"Halaman **{current_page}** akan segera tersedia dengan analisis dan fitur yang lebih mendalam. Untuk saat ini, silakan kembali ke **Dashboard** utama.", icon="🛠️")

       if current_page == "Analytics":
            st.subheader("🔬 Apa yang akan datang di Analisis Lanjutan?")
            st.markdown("- Analisis sentimen per topik secara mendalam")
            st.markdown("- Pemetaan perjalanan pelanggan (Customer Journey Mapping)")
            st.markdown("- Model prediktif untuk churn dan kepuasan")
            st.markdown("- Pembuatan laporan kustom dengan visualisasi interaktif")

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
   main()
