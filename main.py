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
        }

        /* Sidebar Styling */
        .css-1d391kg { /* This class might change with Streamlit versions, be careful */
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0; /* Subtle separator */
            box-shadow: 2px 0 5px rgba(0,0,0,0.05);
        }

        /* Main Content Area */
        .main > div {
            padding: 1.5rem; /* Adjusted padding */
            background-color: #ffffff;
            border-radius: 12px; /* Slightly softer radius */
            margin: 1rem;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); /* Softer shadow */
        }

        /* Metric Cards */
        .metric-card {
            background-color: #ffffff;
            padding: 1.5rem; /* Consistent padding */
            border-radius: 10px;
            border: 1px solid #e0e0e0; /* Subtle border */
            box-shadow: 0 1px 4px rgba(0,0,0,0.05); /* Even softer shadow */
            margin-bottom: 20px;
            transition: box-shadow 0.3s ease;
        }

        .metric-card:hover {
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); /* Slightly more pronounced on hover */
        }

        /* Headers and Titles */
        .dashboard-header {
            text-align: center;
            color: #1a253c; /* Deep navy/charcoal */
            font-size: 2.2rem; /* Slightly adjusted */
            font-weight: 600; /* Bold but not overly so */
            margin-bottom: 1.5rem;
        }

        .section-header {
            font-size: 1.3rem; /* Slightly smaller */
            font-weight: 600;
            color: #1a253c; /* Match dashboard header color */
            /* border-bottom: 2px solid #007bff; */ /* Removing border bottom for a cleaner look */
            padding-bottom: 0.3rem;
            margin: 2.5rem 0 1.2rem 0; /* Adjust margins */
            /* text-transform: uppercase; */ /* Optional: for a more distinct look */
            /* letter-spacing: 0.5px; */ /* Optional */
        }

        /* Buttons */
        .stButton > button {
            background-color: #007bff;
            color: white;
            border-radius: 5px; /* Slightly less rounded for a sharper look */
            border: none;
            padding: 9px 18px; /* Adjusted padding */
            font-weight: 500;
            transition: background-color 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05); /* Subtle shadow for depth */
        }
        .stButton > button:hover {
            background-color: #0056b3;
            box-shadow: 0 2px 4px rgba(0, 91, 179, 0.2); /* Slightly more shadow on hover */
        }

        /* Metric Values */
        .metric-value {
            font-size: 2.5rem; /* Slightly smaller for balance */
            font-weight: bold;
            color: #1a253c; /* Deep navy/charcoal, no gradient */
        }

        .metric-trend-positive {
            color: #28a745; /* Standard success green */
            font-size: 0.9rem;
            font-weight: 600;
        }

        .metric-trend-negative {
            color: #dc3545; /* Standard danger red */
            font-size: 0.9rem;
            font-weight: 600;
        }

        /* Filters */
        .filter-container {
            background-color: #f8f9fa; /* Very light grey */
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            border-left: 3px solid #007bff; /* Accent border */
        }

        /* Chat Messages */
        .stChatMessage[data-testid="stChatMessageContent"] { /* More specific selector */
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }
        .stChatMessage[data-testid="stChatMessageContent"] > div[data-testid="stChatMessageContent"] {
            /* Targeting inner div if necessary, sometimes Streamlit wraps content */
        }

        /* Targeting user and assistant messages based on Streamlit's structure (may need inspection) */
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-user"]) .stChatMessage[data-testid="stChatMessageContent"] {
            background-color: #e7f3ff; /* Light blue for user */
        }
        div[data-testid="stChatMessage"]:has(span[data-testid="chatAvatarIcon-assistant"]) .stChatMessage[data-testid="stChatMessageContent"] {
            background-color: #f1f3f5; /* Light grey for assistant */
        }

        /* Alerts */
        .alert-critical {
            background-color: #f8d7da; /* Bootstrap danger background */
            color: #721c24; /* Bootstrap danger text */
            border: 1px solid #f5c6cb; /* Bootstrap danger border */
            padding: 1rem;
            border-radius: 8px;
            margin: 10px 0;
        }

        .alert-warning {
            background-color: #fff3cd; /* Bootstrap warning background */
            color: #856404; /* Bootstrap warning text */
            border: 1px solid #ffeeba; /* Bootstrap warning border */
            padding: 1rem;
            border-radius: 8px;
            margin: 10px 0;
        }
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
        st.markdown("# 🎯 VOCAL")
        st.markdown("---")
        
        # Navigation Menu
        st.markdown("### 📊 Navigation")
        page = st.selectbox(
            "Choose Page",
            ["Dashboard", "Analytics", "Feedback", "Alerts", "Reports"],
            key="menu_nav"
        )

        st.markdown("---")
        st.markdown("### 👤 User Info")
        st.markdown("**Sebastian**")
        st.markdown("*CX Manager*")
        
        return page

def render_filters(master_df):
    """Render the filter controls"""
    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    st.markdown("### 🔧 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_period = st.selectbox(
            "⏰ Time Period",
            ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
            index=3,
            key="time_filter"
        )
    
    # Get filter options from data
    if not master_df.empty and 'Product' in master_df.columns:
        available_products = sorted(list(master_df['Product'].str.replace("_", " ").str.title().unique()))
    else:
        available_products = ["myBCA", "BCA Mobile", "KPR", "KKB", "KSM", "Investasi", "Asuransi"]
    
    if not master_df.empty and 'Channel' in master_df.columns:
        available_channels = sorted(list(master_df['Channel'].str.replace("_", " ").str.title().unique()))
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
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return time_period, selected_products, selected_channels

def render_health_score_widget(health_data):
    """Render the health score widget"""
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("### 💚 Customer Health Score")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(f'<div class="metric-value">{health_data["score"]}%</div>', unsafe_allow_html=True)

    with col2:
        trend_icon = "📈" if health_data["trend_positive"] else "📉"
        trend_class = "metric-trend-positive" if health_data["trend_positive"] else "metric-trend-negative"
        st.markdown(
           f'<div class="{trend_class}">{trend_icon} {health_data["trend"]} {health_data["trend_label"]}</div>',
           unsafe_allow_html=True
       )

    # Health score chart (Corrected Indentation Starts Here)
    fig_health = create_health_score_chart(health_data)
    st.plotly_chart(fig_health, use_container_width=True, config={'displayModeBar': False})

    # Health score interpretation
    score = health_data["score"]
    if score >= 80:
        st.success("🎉 Excellent customer satisfaction! Keep up the great work.")
    elif score >= 70:
        st.info("👍 Good customer satisfaction with room for improvement.")
    elif score >= 60:
        st.warning("⚠️ Moderate satisfaction. Consider addressing key issues.")
    else:
        st.error("🚨 Low satisfaction detected. Immediate action recommended.")

    st.markdown('</div>', unsafe_allow_html=True) # This closes the metric-card div

def render_alerts_widget():
   """Render the critical alerts widget"""
   st.markdown('<div class="metric-card">', unsafe_allow_html=True)
   st.markdown("### 🚨 Critical Alerts")
      
   # Critical alerts
   st.markdown('<div class="alert-critical">', unsafe_allow_html=True)
   st.markdown("""
   **🔴 Sudden Spike in Negative Sentiment**
   - Mobile App Update X.Y: 45% negative sentiment
   - Volume: 150 mentions in last 3 hours
   - Key Issues: Login failures, App crashes
   - **Action Required**: Immediate technical review
   """)
   st.markdown('</div>', unsafe_allow_html=True)
   
   # High priority alerts
   st.markdown('<div class="alert-warning">', unsafe_allow_html=True)
   st.markdown("""
   **🟡 High Churn Risk Pattern**
   - Pattern: Repeated billing errors in Savings accounts
   - Affected: 12 unique customer patterns identified
   - Average sentiment: -0.8 (Very Negative)
   - **Action**: Customer retention outreach recommended
   """)
   st.markdown('</div>', unsafe_allow_html=True)
   
   col1, col2 = st.columns(2)
   with col1:
       if st.button("🔍 View All Alerts", type="primary"):
           st.info("Redirecting to detailed alerts page...")
   
   with col2:
       if st.button("📋 Create Action Plan", type="secondary"):
           st.info("Opening action plan creator...")
   
   st.markdown('</div>', unsafe_allow_html=True)

def render_hotspots_widget():
   """Render the predictive hotspots widget"""
   st.markdown('<div class="metric-card">', unsafe_allow_html=True)
   st.markdown("### 🔮 Predictive Hotspots")
      
   # Emerging issues
   st.markdown("**🆕 New Overdraft Policy Confusion**")
   st.markdown("- Impact Level: Medium 🟡")
   st.markdown("- 'Confused' language patterns: +30% WoW")
   st.markdown("- Common phrases: 'don't understand', 'how it works'")
   st.markdown("- **Recommendation**: Create clearer policy explanation")
   
   st.markdown("---")
   
   st.markdown("**🌐 International Transfer UI Issues**")
   st.markdown("- Impact Level: Low 🟢")
   st.markdown("- Task abandonment rate: +15% MoM")
   st.markdown("- Negative sentiment around 'Beneficiary Setup'")
   st.markdown("- **Recommendation**: UI/UX review for international transfers")
   
   # Trend indicators
   col1, col2, col3 = st.columns(3)
   with col1:
       st.metric("Emerging Issues", "2", delta="1")
   with col2:
       st.metric("Trending Topics", "5", delta="2")
   with col3:
       st.metric("Predicted Risks", "3", delta="-1")
   
   st.markdown('</div>', unsafe_allow_html=True)

def render_voice_snapshot(analytics_data, time_period):
   """Render the customer voice snapshot section"""
   st.markdown('<div class="section-header">📊 Customer Voice Snapshot</div>', unsafe_allow_html=True)
      
   col1, col2, col3 = st.columns(3)
   
   # Sentiment Distribution
   with col1:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### 😊 Sentiment Distribution")
       
       fig_sentiment = create_sentiment_chart(analytics_data['sentiment_data'])
       st.plotly_chart(fig_sentiment, use_container_width=True, config={'displayModeBar': False})
       
       # Add sentiment insights
       sentiment_summary = analytics_data['sentiment_summary']
       if 'Positif' in sentiment_summary:
           st.success(f"Positive sentiment: {sentiment_summary['Positif'].split('(')[0]}")
       if 'Negatif' in sentiment_summary:
           st.error(f"Negative sentiment: {sentiment_summary['Negatif'].split('(')[0]}")
       
       st.markdown('</div>', unsafe_allow_html=True)
   
   # Intent Distribution
   with col2:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### 🎯 Intent Distribution (Top 5)")
       
       fig_intent = create_intent_chart(analytics_data['intent_data'])
       st.plotly_chart(fig_intent, use_container_width=True, config={'displayModeBar': False})
       
       # Add intent insights
       intent_summary = analytics_data['intent_summary']
       if intent_summary and "Info" not in intent_summary:
           top_intent = list(intent_summary.keys())[0]
           st.info(f"Top intent: {top_intent}")
       
       st.markdown('</div>', unsafe_allow_html=True)
   
   # Volume Trend
   with col3:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown(f"### 📈 Volume Trend ({time_period})")
       
       fig_volume = create_volume_chart(analytics_data['volume_data'])
       st.plotly_chart(fig_volume, use_container_width=True, config={'displayModeBar': False})
       
       # Add volume insights
       total_interactions = analytics_data['total_interactions']
       st.metric("Total Interactions", total_interactions)
       
       st.markdown('</div>', unsafe_allow_html=True)

def render_customer_themes():
   """Render the customer themes section"""
   st.markdown('<div class="section-header">💭 Top Customer Themes</div>', unsafe_allow_html=True)
   
   col1, col2 = st.columns(2)
   
   with col1:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### 🌟 Top Positive Themes")
       
       positive_themes = [
           {"theme": "Fast Customer Service", "mentions": 156, "sentiment": 0.85},
           {"theme": "Easy Mobile Banking", "mentions": 134, "sentiment": 0.82},
           {"theme": "Helpful Staff", "mentions": 112, "sentiment": 0.78},
           {"theme": "Quick Problem Resolution", "mentions": 98, "sentiment": 0.80},
           {"theme": "User-Friendly Interface", "mentions": 87, "sentiment": 0.77}
       ]
       
       for theme in positive_themes:
           st.markdown(f"**{theme['theme']}**")
           st.markdown(f"- {theme['mentions']} mentions")
           st.markdown(f"- Sentiment: {theme['sentiment']:.2f}")
           st.markdown("---")
       
       st.success('💬 "Support resolved my issue in minutes! So efficient and professional."')
       st.markdown('</div>', unsafe_allow_html=True)
   
   with col2:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### ⚠️ Top Negative Themes")
       
       negative_themes = [
           {"theme": "App Technical Issues", "mentions": 89, "sentiment": -0.72},
           {"theme": "Long Wait Times", "mentions": 76, "sentiment": -0.68},
           {"theme": "Fee Transparency", "mentions": 65, "sentiment": -0.58},
           {"theme": "Login Problems", "mentions": 54, "sentiment": -0.65},
           {"theme": "Complex Procedures", "mentions": 43, "sentiment": -0.55}
       ]
       
       for theme in negative_themes:
           st.markdown(f"**{theme['theme']}**")
           st.markdown(f"- {theme['mentions']} mentions")
           st.markdown(f"- Sentiment: {theme['sentiment']:.2f}")
           st.markdown("---")
       
       st.error('💬 "The app keeps crashing after the latest update. Very frustrating experience."')
       st.markdown('</div>', unsafe_allow_html=True)

def render_opportunity_radar():
   """Render the opportunity radar section"""
   st.markdown('<div class="section-header">🎯 Opportunity Radar</div>', unsafe_allow_html=True)
  
   col1, col2, col3 = st.columns(3)
   
   with col1:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### 🎉 Delightful Features")
       st.markdown("""
       **Instant Card Activation**
       - 75 delight mentions this week
       - Sentiment Score: +0.95 (Exceptional)
       - Keywords: "amazing", "so easy", "instant"
       - **Opportunity**: Amplify in marketing campaigns
       - **ROI Potential**: High
       """)
       
       st.success("**Action**: Showcase in customer testimonials")
       st.markdown('</div>', unsafe_allow_html=True)
   
   with col2:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### 💰 Cross-Sell Opportunities")
       st.markdown("""
       **Mortgage Inquiry Surge**
       - Mortgage info requests: +15% WoW
       - Related topics: Savings, Financial Planning
       - Customer segments: 25-40 age group
       - **Opportunity**: Targeted mortgage promotions
       - **ROI Potential**: Very High
       """)
       
       st.info("**Action**: Create personalized mortgage offers")
       st.markdown('</div>', unsafe_allow_html=True)
   
   with col3:
       st.markdown('<div class="metric-card">', unsafe_allow_html=True)
       st.markdown("### ⭐ Service Excellence")
       st.markdown("""
       **Complex Issue Resolution**
       - 25 excellence mentions for complex cases
       - Top performers: Agent A, B, C
       - Resolution time: 15% faster than average
       - **Opportunity**: Scale best practices
       - **ROI Potential**: Medium-High
       """)
       
       st.success("**Action**: Implement training program")
       st.markdown('</div>', unsafe_allow_html=True)

def render_vira_chat(dashboard_state):
   """Render the VIRA chat interface"""
   st.markdown('<div class="section-header">🤖 Chat with VIRA</div>', unsafe_allow_html=True)
   
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
       # Add user message
       st.session_state.messages.append({"role": "user", "content": prompt})
       with st.chat_message("user"):
           st.markdown(prompt)
       
       # Generate AI response
       with st.chat_message("assistant"):
           message_placeholder = st.empty()
           full_response = ""
           
           try:
               # Show typing indicator
               with st.spinner("VIRA sedang menganalisis data..."):
                   stream = generate_llm_response(prompt, dashboard_state, SYSTEM_PROMPT_VIRA)
                   
                   for chunk in stream:
                       full_response += chunk
                       message_placeholder.markdown(full_response + "▌")
                   
                   message_placeholder.markdown(full_response)
               
           except Exception as e:
               error_message = f"🚫 Maaf, terjadi kesalahan: {str(e)}. Silakan coba lagi dalam beberapa saat."
               message_placeholder.error(error_message)
               full_response = error_message
       
       # Add assistant response to history
       st.session_state.messages.append({"role": "assistant", "content": full_response})
   
   # Chat controls
   col1, col2, col3 = st.columns([1, 1, 1])
   
   with col1:
       if st.button("🔄 Clear Chat", type="secondary"):
           st.session_state.messages = [
               {
                   "role": "assistant",
                   "content": "Chat history cleared. How can I help you today?"
               }
           ]
           st.rerun()
   
   with col2:
       if st.button("💾 Export Chat", type="secondary"):
           chat_export = "\n".join([
               f"{msg['role'].title()}: {msg['content']}"
               for msg in st.session_state.messages
           ])
           st.download_button(
               "📥 Download Chat",
               chat_export,
               "vira_chat_export.txt",
               "text/plain"
           )
   
   with col3:
       if st.button("❓ Chat Help", type="secondary"):
           st.info("""
           **VIRA dapat membantu dengan:**
           - Analisis tren sentimen dan volume
           - Interpretasi data dasbor
           - Identifikasi pola dan anomali
           - Rekomendasi aksi berdasarkan data
           - Perbandingan metrik antar periode
           """)

# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
   """Main application function"""
   # Apply custom styling
   apply_custom_css()
   
   # Load data
   with st.spinner("🔄 Loading data from Google Sheets..."):
       master_df = load_data_from_google_sheets()
   
   # Render sidebar and get page selection
   current_page = render_sidebar()
   
   # Main content based on page selection
   if current_page == "Dashboard":
       # Dashboard header
       st.markdown('<div class="dashboard-header">🏦 Voice of Customer Dashboard</div>', unsafe_allow_html=True)
       st.markdown("*Real-time Customer Experience Insights & Performance Analytics*")
       
       # Render filters
       time_period, selected_products, selected_channels = render_filters(master_df)
       
       # Apply filters to data
       filtered_df = master_df.copy()
       filtered_df = apply_time_filter(filtered_df, time_period)
       filtered_df = apply_product_filter(filtered_df, selected_products)
       filtered_df = apply_channel_filter(filtered_df, selected_channels)
       
       # Process analytics data
       analytics_data = process_filtered_data(filtered_df)
       
       # Get health score data
       health_score_data = generate_health_score_data()
       time_period_map = {
           "All Periods": "all", "Today": "today", "This Week": "week",
           "This Month": "month", "This Quarter": "quarter", "This Year": "year"
       }
       selected_time_key = time_period_map.get(time_period, "month")
       current_health_data = health_score_data[selected_time_key].copy()
       current_health_data['time_period_label'] = time_period
       
       # Dashboard widgets section
       st.markdown('<div class="section-header">📊 Dashboard Widgets</div>', unsafe_allow_html=True)
       
       col1, col2, col3 = st.columns(3)
       
       with col1:
           render_health_score_widget(current_health_data)
       
       with col2:
           render_alerts_widget()
       
       with col3:
           render_hotspots_widget()
       
       # Customer voice snapshot
       render_voice_snapshot(analytics_data, time_period)
       
       # Generate summary insights
       if not filtered_df.empty:
           summary_parts = []
           sentiment_summary = analytics_data['sentiment_summary']
           intent_summary = analytics_data['intent_summary']
           
           if 'Positif' in sentiment_summary:
               positive_pct = sentiment_summary['Positif'].split('%')[0]
               summary_parts.append(f"Sentimen positif mendominasi ({positive_pct}%)")
           
           if intent_summary and "Info" not in intent_summary:
               top_intent = list(intent_summary.keys())[0]
               summary_parts.append(f"intent '{top_intent}' paling sering muncul")
           
           if analytics_data['total_interactions'] > 0:
               summary_parts.append(f"dengan total {analytics_data['total_interactions']} interaksi")
           
           if summary_parts:
               st.info(f"📈 **Ringkasan**: {', '.join(summary_parts)} dalam periode {time_period.lower()}.")
       else:
           st.warning("⚠️ Tidak ada data yang tersedia untuk filter yang dipilih.")
       
       # Customer themes
       render_customer_themes()
       
       # Opportunity radar
       render_opportunity_radar()
       
       # Prepare dashboard state for VIRA
       dashboard_state = {
           **current_health_data,
           "time_period_label_llm": time_period,
           'total_interactions': analytics_data.get('total_interactions', 'N/A'),
           'sentiment_summary': analytics_data.get('sentiment_summary', {}),
           'intent_summary': analytics_data.get('intent_summary', {}),
           'volume_summary': analytics_data.get('volume_summary', 'N/A'),
           # Update VIRA's knowledge about alerts/hotspots
           'critical_alerts_summary': "The dashboard displays illustrative examples of critical alerts.",
           'emerging_hotspots_summary': "The dashboard shows example emerging customer hotspots."
       }

       
       # VIRA Chat
       render_vira_chat(dashboard_state)
       
   else:
       # Other pages (placeholder)
       st.markdown('<div class="dashboard-header">🚧 Coming Soon</div>', unsafe_allow_html=True)
       st.markdown(f"## {current_page}")
       st.info(f"Halaman {current_page} sedang dalam pengembangan. Silakan pilih 'Dashboard' untuk melihat dashboard utama.")
       
       # Add some placeholder content based on page
       if current_page == "Analytics":
           st.markdown("### 📊 Advanced Analytics")
           st.markdown("- Detailed sentiment analysis")
           st.markdown("- Customer journey mapping")
           st.markdown("- Predictive modeling")
           st.markdown("- Custom report generation")
       
       elif current_page == "Feedback":
           st.markdown("### 💬 Customer Feedback")
           st.markdown("- Feedback collection management")
           st.markdown("- Response tracking")
           st.markdown("- Satisfaction surveys")
           st.markdown("- Feedback categorization")
       
       elif current_page == "Alerts":
           st.markdown("### 🚨 Alert Management")
           st.markdown("- Real-time alert configuration")
           st.markdown("- Escalation procedures")
           st.markdown("- Alert history")
           st.markdown("- Performance monitoring")
       
       elif current_page == "Reports":
           st.markdown("### 📈 Reporting")
           st.markdown("- Executive dashboards")
           st.markdown("- Scheduled reports")
           st.markdown("- Custom analytics")
           st.markdown("- Data export capabilities")

# ==============================================================================
# APPLICATION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
   main()
