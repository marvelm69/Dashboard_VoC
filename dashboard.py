import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="Voice of Customer Dashboard - Refined Aesthetics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for styling
st.markdown("""
<style>
    /* Main styles */
    .stApp {
        background: linear-gradient(170deg, #fbfcfe 0%, #f2f4f6 100%);
    }

    /* Container styles */
    .css-1d391kg, .css-12oz5g7 {
        max-width: 1600px;
        padding: 1rem;
    }

    /* Card styling */
    .card {
        background: white;
        border-radius: 14px;
        padding: 1.2rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.03), 0 5px 10px rgba(0, 0, 0, 0.06);
        margin-bottom: 1.2rem;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }

    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.03), 0 8px 18px rgba(0, 0, 0, 0.06);
    }

    .card-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 0.8rem;
        border-bottom: 1px solid #e5e5ea;
        padding-bottom: 0.8rem;
    }

    .card-title {
        font-size: 1.05rem;
        font-weight: 500;
        color: #1d1d1f;
        letter-spacing: -0.01em;
        margin: 0;
    }

    /* Health score styling */
    .health-score {
        text-align: center;
        margin: 0.8rem 0;
    }

    .health-score-value {
        font-size: 3rem;
        font-weight: 500;
        color: #007aff;
    }

    .health-trend {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.4rem;
        color: #34c759;
        font-weight: 400;
        font-size: 0.85rem;
    }

    .negative {
        color: #ff3b30;
    }

    /* Alert styling */
    .alert-item {
        display: flex;
        align-items: flex-start;
        gap: 0.8rem;
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-radius: 10px;
        background: white;
        border-left: 4px solid;
    }

    .alert-critical {
        border-left-color: #ff3b30;
    }

    .alert-high {
        border-left-color: #ff9500;
    }

    .alert-icon {
        width: 0.6rem;
        height: 0.6rem;
        border-radius: 50%;
        flex-shrink: 0;
        margin-top: 0.3rem;
    }

    .alert-critical .alert-icon {
        background: #ff3b30;
    }

    .alert-high .alert-icon {
        background: #ff9500;
    }

    /* Theme styling */
    .theme-item {
        padding: 0.6rem 0.8rem;
        margin-bottom: 0.6rem;
        border-radius: 10px;
        font-size: 0.78rem;
        border: 1px solid transparent;
    }

    .positive-theme {
        background: rgba(52, 199, 89, 0.12);
        color: rgba(52, 199, 89, 0.88);
        border-color: rgba(52, 199, 89, 0.25);
    }

    .negative-theme {
        background: rgba(255, 59, 48, 0.12);
        color: rgba(255, 59, 48, 0.88);
        border-color: rgba(255, 59, 48, 0.25);
    }

    .quote-item {
        padding: 0.8rem;
        margin-top: 0.8rem;
        border-radius: 10px;
        background: rgba(0, 0, 0, 0.02);
        font-size: 0.78rem;
        color: #4a4a4f;
        border: 1px solid #e5e5ea;
        font-style: italic;
        line-height: 1.4;
    }

    /* Hotspot styling */
    .hotspot-item {
        padding: 0.8rem;
        margin-bottom: 0.8rem;
        border-radius: 10px;
        background: white;
        border: 1px solid #e5e5ea;
    }

    .hotspot-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.4rem;
    }

    .impact-indicator {
        padding: 0.25rem 0.6rem;
        border-radius: 14px;
        font-size: 0.7rem;
        font-weight: 500;
    }

    .impact-medium {
        background: rgba(255, 149, 0, 0.18);
        color: rgba(255, 149, 0, 0.85);
    }

    .impact-low {
        background: rgba(52, 199, 89, 0.18);
        color: rgba(52, 199, 89, 0.85);
    }

    /* Opportunity styling */
    .opportunity-item {
        padding: 0.8rem;
        border-radius: 10px;
        background: white;
        border: 1px solid #e5e5ea;
        margin-bottom: 0.8rem;
    }

    /* Widget summary styling */
    .widget-summary {
        padding: 0.8rem;
        background: rgba(0, 0, 0, 0.02);
        border-radius: 10px;
        font-size: 0.85rem;
        color: #4a4a4f;
        border: 1px solid #e5e5ea;
        line-height: 1.45;
        margin-top: 0.8rem;
    }

    /* Filter button styling */
    .filter-btn {
        display: inline-block;
        padding: 0.5rem 1rem;
        border: 1px solid #d2d2d7;
        border-radius: 14px;
        background: white;
        font-size: 0.78rem;
        font-weight: 400;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #4a4a4f;
        margin-right: 0.6rem;
        margin-bottom: 0.6rem;
    }

    .filter-btn-active {
        background: #007aff;
        color: white;
        border-color: #007aff;
        font-weight: 500;
    }

    /* Layout adjustments for Streamlit */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 10px;
        padding: 8px 16px;
        border: 1px solid #d2d2d7;
        color: #4a4a4f;
    }

    .stTabs [aria-selected="true"] {
        background-color: #007aff !important;
        color: white !important;
        border-color: #007aff !important;
        font-weight: 500;
    }

    /* Add spacing to selectbox */
    div[data-baseweb="select"] {
        margin-bottom: 1rem;
    }

    /* For multiselect styles */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #f0f2f5;
        border-radius: 7px;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Mock data for dashboard
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

# Sidebar with navigation
with st.sidebar:
    st.image("https://via.placeholder.com/50x50.png?text=VOCAL", width=50)
    st.markdown("## VOCAL")
    st.markdown("---")

    selected = option_menu(
        "Menu", 
        ["Dashboard", "Analytics", "Feedback", "Alerts", "Reports"],
        icons=["grid", "graph-up", "chat", "exclamation-triangle", "clipboard"], 
        menu_icon="list", 
        default_index=0,
    )

    st.markdown("### Customer Insights")
    insights = option_menu(
        "", 
        ["Sentiment Analysis", "Journey Mapping", "Satisfaction Scores", "Theme Analysis"],
        icons=["emoji-smile", "bullseye", "bar-chart", "search"], 
        menu_icon=None, 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#4a4a4f", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "padding": "10px 20px"},
            "nav-link-selected": {"background-color": "#007aff", "font-weight": "normal"},
        }
    )

    st.markdown("### Operations")
    operations = option_menu(
        "", 
        ["Real-time Monitoring", "Predictive Analytics", "Performance Metrics", "Action Items"],
        icons=["lightning", "magic", "bar-chart", "bullseye"], 
        menu_icon=None, 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#4a4a4f", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "left", "margin":"0px", "padding": "10px 20px"},
            "nav-link-selected": {"background-color": "#007aff", "font-weight": "normal"},
        }
    )

    # User info at bottom of sidebar
    st.markdown("---")
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("""
        <div style="width: 40px; height: 40px; border-radius: 50%; background: linear-gradient(145deg, #007aff 0%, #005ecb 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: 500;">SB</div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("**Sebastian**")
        st.markdown("<span style='color: #4a4a4f; font-size: 0.8rem;'>CX Manager</span>", unsafe_allow_html=True)

# App header
st.markdown("<h1 style='font-size: 1.7rem; font-weight: 600; margin-bottom: 0.2rem;'>Customer Experience Health</h1>", unsafe_allow_html=True)
st.markdown("<p style='font-size: 0.95rem; color: #4a4a4f; margin-bottom: 1.5rem;'>Real-time Insights & Performance Overview</p>", unsafe_allow_html=True)

# Filter section
col1, col2, col3 = st.columns(3)

with col1:
    time_period = st.selectbox(
        "TIME",
        ["All Periods", "Today", "This Week", "This Month", "This Quarter", "This Year"],
        index=3
    )

with col2:
    products = st.multiselect(
        "PRODUCT",
        ["All Products", "myBCA", "BCA Mobile", "KPR", "KKB", "KSM", "Investasi", "Asuransi", "KMK", "Kartu Kredit", "EDC & QRIS", "Poket Valas"],
        default=["All Products"]
    )

with col3:
    channels = st.multiselect(
        "CHANNEL",
        ["All Channels", "Social Media", "Call Center", "WhatsApp", "Webchat", "VIRA", "E-mail", "Survey Gallup", "Survey BSQ", "Survey CX"],
        default=["All Channels"]
    )

# Convert filter selections to internal format
time_period_map = {
    "All Periods": "all",
    "Today": "today",
    "This Week": "week",
    "This Month": "month",
    "This Quarter": "quarter",
    "This Year": "year"
}
selected_time = time_period_map.get(time_period, "month")

# Handle "All Products" selection
if "All Products" in products:
    product_filter = ["all"]
else:
    product_filter = [p.lower().replace(" ", "_") for p in products]

# Handle "All Channels" selection
if "All Channels" in channels:
    channel_filter = ["all"]
else:
    channel_filter = [c.lower().replace(" ", "_") for c in channels]

# Get health score data based on filters
health_score_data = generate_health_score_data()
current_health_data = health_score_data.get(selected_time, health_score_data["month"])

# Adjust other metrics based on filters
product_multiplier = 1.0 if "all" in product_filter else 0.8
channel_multiplier = 1.0 if "all" in channel_filter else 0.9
if "social_media" in channel_filter and "all" not in channel_filter:
    channel_multiplier = 0.6

# Main dashboard layout with 3 columns
col1, col2, col3 = st.columns(3)

# Column 1: Customer Health Score
with col1:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Customer Health Score</h3>
            <div>
                <button class="filter-btn">Export</button>
            </div>
        </div>
        <div>
            <button class="filter-btn filter-btn-active">Real-time</button>
            <button class="filter-btn">Daily Trend</button>
            <button class="filter-btn">Comparison</button>
        </div>
        <div class="health-score">
            <div class="health-score-value">{score}<span style="font-size: 1.8rem; color: #4a4a4f;">%</span></div>
            <div class="health-trend {trend_class}">
                <span>{trend_icon}</span> <span>{trend} {trend_label}</span>
            </div>
        </div>
    """.format(
        score=current_health_data["score"],
        trend=current_health_data["trend"],
        trend_label=current_health_data["trend_label"],
        trend_class="" if current_health_data["trend_positive"] else "negative",
        trend_icon="↑" if current_health_data["trend_positive"] else "↓"
    ), unsafe_allow_html=True)

    # Health trend chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=current_health_data["labels"],
        y=current_health_data["values"],
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(52,199,89,0.18)',
        line=dict(color='#34c759', width=2),
        name='Health Score'
    ))

    fig.update_layout(
        height=150,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickfont=dict(
                color='#4a4a4f',
                size=9
            )
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#e5e5ea',
            showline=False,
            showticklabels=True,
            tickfont=dict(
                color='#4a4a4f',
                size=9
            ),
            range=[min(current_health_data["values"]) - 2, max(current_health_data["values"]) + 2]
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    st.markdown("""
        <div class="widget-summary">
            Overall customer satisfaction is strong, showing a positive trend this month.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Column 2: Critical Alerts
with col2:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Critical Alerts</h3>
            <div>
                <button class="filter-btn" style="background: linear-gradient(180deg, #007aff 0%, #005ecb 100%); color: white; border: none;">Acknowledge All</button>
            </div>
        </div>
        <div>
            <button class="filter-btn filter-btn-active">Critical</button>
            <button class="filter-btn">High</button>
            <button class="filter-btn">Medium</button>
            <button class="filter-btn">All</button>
        </div>
        <div class="alert-item alert-critical">
            <div class="alert-icon" style="background: #ff3b30; width: 0.6rem; height: 0.6rem; border-radius: 50%; flex-shrink: 0; margin-top: 0.3rem;"></div>
            <div>
                <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.25rem; color: #1d1d1f;">Sudden Spike in Negative Sentiment</h4>
                <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
                    Mobile App Update X.Y: 45% negative<br>
                    Volume: 150 mentions / 3 hrs<br>
                    Issues: Login Failed, App Crashing
                </div>
            </div>
        </div>
        <div class="alert-item alert-high">
            <div class="alert-icon" style="background: #ff9500; width: 0.6rem; height: 0.6rem; border-radius: 50%; flex-shrink: 0; margin-top: 0.3rem;"></div>
            <div>
                <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.25rem; color: #1d1d1f;">High Churn Risk Pattern Detected</h4>
                <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
                    Pattern: Repeated Billing Errors - Savings<br>
                    12 unique customer patterns<br>
                    Avg. sentiment: -0.8
                </div>
            </div>
        </div>
        <button style="background: linear-gradient(180deg, #007aff 0%, #005ecb 100%); color: white; border: none; padding: 0.7rem 1.2rem; border-radius: 10px; font-weight: 400; cursor: pointer; width: 100%; font-size: 0.9rem; margin-top: 0.8rem;">View All Alerts</button>
    </div>
    """, unsafe_allow_html=True)

# Column 3: Predictive Hotspots
with col3:
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h3 class="card-title">Predictive Hotspots</h3>
            <div>
                <button class="filter-btn" style="background: linear-gradient(180deg, #007aff 0%, #005ecb 100%); color: white; border: none;">Create Action</button>
            </div>
        </div>
        <div>
            <button class="filter-btn filter-btn-active">Emerging</button>
            <button class="filter-btn">Trending</button>
            <button class="filter-btn">Predicted</button>
        </div>
        <div class="hotspot-item">
            <div class="hotspot-header">
                <h4 style="font-size: 0.95rem; font-weight: 500;">New Overdraft Policy Confusion</h4>
                <span class="impact-indicator impact-medium">Medium Impact</span>
            </div>
            <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
                'Confused' Language: +30% WoW<br>
                Keywords: "don't understand", "how it works"
            </div>
        </div>
        <div class="hotspot-item">
            <div class="hotspot-header">
                <h4 style="font-size: 0.95rem; font-weight: 500;">Intl. Transfer UI Issues</h4>
                <span class="impact-indicator impact-low">Low Impact</span>
            </div>
            <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
                Task Abandonment: +15% MoM<br>
                Negative sentiment: 'Beneficiary Setup'
            </div>
        </div>
        <div class="widget-summary">
            Monitor emerging confusion on overdrafts and usability for international transfers.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Customer Voice Snapshot - Full width
st.markdown("""
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Customer Voice Snapshot</h3>
        <div>
            <button class="filter-btn">Drill Down</button>
            <button class="filter-btn">Export</button>
        </div>
    </div>
    <div>
        <button class="filter-btn filter-btn-active">Overview</button>
        <button class="filter-btn">Sentiment</button>
        <button class="filter-btn">Intent</button>
        <button class="filter-btn">Volume</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Charts for Customer Voice Snapshot
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<h4 style='margin-bottom: 0.8rem; font-weight: 400; font-size: 0.95rem; color: #4a4a4f;'>Sentiment Distribution</h4>", unsafe_allow_html=True)

    # Sentiment distribution data adjusted by product_multiplier
    sentiment_data = pd.DataFrame({
        'Category': ['Positive', 'Neutral', 'Negative'],
        'Value': [
            (60 + random.random() * 10) * product_multiplier,
            (20 + random.random() * 5) * product_multiplier,
            (10 + random.random() * 5) * product_multiplier
        ]
    })

    fig = px.pie(
        sentiment_data, 
        values='Value', 
        names='Category',
        color='Category',
        color_discrete_map={
            'Positive': '#34c759',
            'Neutral': '#a2a2a7',
            'Negative': '#ff3b30'
        },
        hole=0.75
    )

    fig.update_layout(
        height=230,
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5,
            font=dict(size=10)
        ),
        showlegend=True
    )

    fig.update_traces(textinfo='percent', textfont_size=10)

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col2:
    st.markdown("<h4 style='margin-bottom: 0.8rem; font-weight: 400; font-size: 0.95rem; color: #4a4a4f;'>Intent Distribution</h4>", unsafe_allow_html=True)

    # Intent distribution data
    intent_data = pd.DataFrame({
        'Intent': ['Info Seeking', 'Complaint', 'Service Request', 'Feedback'],
        'Value': [
            35 + random.random() * 10,
            20 + random.random() * 5,
            20 + random.random() * 5,
            10 + random.random() * 5
        ]
    })

    fig = px.bar(
        intent_data,
        y='Intent',
        x='Value',
        orientation='h',
        color='Intent',
        color_discrete_map={
            'Info Seeking': '#007aff',
            'Complaint': '#ff9500',
            'Service Request': '#5856d6',
            'Feedback': '#ffcc00'
        }
    )

    fig.update_layout(
        height=230,
        margin=dict(l=0, r=10, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='#e5e5ea',
            showline=False,
            showticklabels=True
        ),
        yaxis=dict(
            title=None,
            showgrid=False,
            showline=False,
            showticklabels=True
        ),
        showlegend=False
    )

    fig.update_traces(
        marker_line_width=0,
        marker_line_color='rgba(0,0,0,0)',
        width=0.6
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

with col3:
    st.markdown("<h4 style='margin-bottom: 0.8rem; font-weight: 400; font-size: 0.95rem; color: #4a4a4f;'>Volume Trend (30 Days)</h4>", unsafe_allow_html=True)

    # Volume trend data adjusted by channel_multiplier
    days = list(range(1, 31))
    volume_data = [(400 + random.random() * 300 + i * 5) * channel_multiplier for i in range(30)]

    vol_df = pd.DataFrame({
        'Day': days,
        'Volume': volume_data
    })

    fig = px.line(
        vol_df, 
        x='Day', 
        y='Volume',
        line_shape='spline'
    )

    fig.update_traces(
        line_color='#007aff',
        fill='tozeroy',
        fillcolor='rgba(0,122,255,0.18)',
        mode='lines'
    )

    fig.update_layout(
        height=230,
        margin=dict(l=0, r=10, t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            title=None,
            showgrid=False,
            showline=False,
            showticklabels=True,
            tickmode='array',
            tickvals=[1, 5, 10, 15, 20, 25, 30],
            tickfont=dict(size=9)
        ),
        yaxis=dict(
            title=None,
            showgrid=True,
            gridcolor='#e5e5ea',
            showline=False,
            showticklabels=True,
            tickfont=dict(size=9),
            range=[min(volume_data) - 20, max(volume_data) + 20]
        )
    )

    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# Summary under charts
st.markdown("""
<div class="widget-summary" style="margin-top: 0;">
    Positive sentiment leads at 65%. Information-seeking is top intent (40%). Volume shows steady increase.
</div>
""", unsafe_allow_html=True)

# Top Customer Themes - Full width
st.markdown("""
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Top Customer Themes</h3>
        <div>
            <button class="filter-btn" style="background: linear-gradient(180deg, #007aff 0%, #005ecb 100%); color: white; border: none;">Analyze Themes</button>
        </div>
    </div>
    <div>
        <button class="filter-btn filter-btn-active">Top 10</button>
        <button class="filter-btn">Trending</button>
        <button class="filter-btn">Emerging</button>
        <button class="filter-btn">Declining</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Two columns for positive and negative themes
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h4 style='color: #34c759; padding-bottom: 0.5rem;'>Top Positive Themes</h4>", unsafe_allow_html=True)
    st.markdown("<div class='theme-item positive-theme'>Fast Customer Service</div>", unsafe_allow_html=True)
    st.markdown("<div class='theme-item positive-theme'>Easy Mobile Banking</div>", unsafe_allow_html=True)
    st.markdown("<div class='theme-item positive-theme'>Helpful Staff</div>", unsafe_allow_html=True)
    st.markdown("<div class='quote-item'>\"Support resolved my issue in minutes! So efficient.\"</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<h4 style='color: #ff3b30; padding-bottom: 0.5rem;'>Top Negative Themes</h4>", unsafe_allow_html=True)
    st.markdown("<div class='theme-item negative-theme'>App Technical Issues</div>", unsafe_allow_html=True)
    st.markdown("<div class='theme-item negative-theme'>Long Wait Times (Call)</div>", unsafe_allow_html=True)
    st.markdown("<div class='theme-item negative-theme'>Fee Transparency</div>", unsafe_allow_html=True)
    st.markdown("<div class='quote-item'>\"The app keeps crashing after the latest update. Very frustrating.\"</div>", unsafe_allow_html=True)

# Opportunity Radar - Full width
st.markdown("""
<div class="card">
    <div class="card-header">
        <h3 class="card-title">Opportunity Radar</h3>
        <div>
            <button class="filter-btn" style="background: linear-gradient(180deg, #007aff 0%, #005ecb 100%); color: white; border: none;">Prioritize</button>
        </div>
    </div>
    <div>
        <button class="filter-btn filter-btn-active">High Value</button>
        <button class="filter-btn">Quick Wins</button>
        <button class="filter-btn">Strategic</button>
    </div>
</div>
""", unsafe_allow_html=True)

# Three columns for opportunities
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="opportunity-item">
        <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">🎉 Delightful: Instant Card Activation</h4>
        <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
            75 delight mentions this week (Sentiment: +0.95)<br>
            Keywords: "amazing", "so easy", "instant"<br>
            <strong>Action:</strong> Amplify in marketing? Benchmark?
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="opportunity-item">
        <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">💰 Cross-Sell: Mortgage Inquiries +15%</h4>
        <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
            Mortgage info seeking: +15% WoW<br>
            Related: Savings, Financial Planning<br>
            <strong>Action:</strong> Target with relevant mortgage info?
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="opportunity-item">
        <h4 style="font-size: 0.95rem; font-weight: 500; margin-bottom: 0.4rem;">⭐ Service Excellence: Complex Issues</h4>
        <div style="font-size: 0.78rem; color: #4a4a4f; line-height: 1.5;">
            25 positive mentions for complex issue resolution<br>
            Agents: A, B, C praised.<br>
            <strong>Action:</strong> Identify best practices? Recognize agents?
        </div>
    </div>
    """, unsafe_allow_html=True)

# AI Assistant - Chat widget
st.markdown("""
<div style="position: fixed; bottom: 20px; right: 20px; z-index: 1000;">
    <div style="position: relative;">
        <div id="chat-container" style="display: none; position: absolute; bottom: 60px; right: 0; width: 360px; height: 500px; background: white; border-radius: 14px; box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12); overflow: hidden; flex-direction: column; border: 1px solid #d2d2d7;">
            <div style="background: #f8f8fa; color: #1d1d1f; padding: 0.8rem; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #e5e5ea;">
                <h3 style="font-size: 1rem; font-weight: 500; margin: 0;">VIRA</h3>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Create a container for the chat widget
chat_container = st.container()

# Add chat widget with expander for clean UI
with st.expander("💬 Chat with VIRA", expanded=False):
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm VIRA your AI assistant. How can I help with the dashboard today?"}
        ]

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Function for generating responses
    def generate_response(prompt):
        # Simple response logic based on user input
        prompt_lower = prompt.lower()

        if "health score" in prompt_lower:
            return "Customer Health Score is 82%, up 1.5% from last month. More details?"

        elif "alerts" in prompt_lower:
            return "2 critical alerts: Mobile app sentiment spike & Churn risk from billing errors. Details or actions?"

        elif "hotspots" in prompt_lower:
            return "Hotspots: Overdraft policy confusion (medium impact) & Intl. transfer UI issues (low impact). Explore further?"

        elif "opportunities" in prompt_lower:
            return "Opportunities: Promote instant card activation, target mortgage inquiries, scale service excellence. Interested in one?"

        elif "thank" in prompt_lower:
            return "You're welcome! Anything else?"

        else:
            return 'I can help with dashboard insights. Try "health score trends", "summarize alerts", or "top opportunities".'

    # Accept user input
    if prompt := st.chat_input("Ask about insights, alerts..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})