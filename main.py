import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import random
import uuid

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
            index=3,
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
    selected_time = time_period_map.get(time_period, "month")
    product_filter = ["all"] if "All Products" in products else [p.lower().replace(" ", "_") for p in products]
    channel_filter = ["all"] if "All Channels" in channels else [c.lower().replace(" ", "_") for c in channels]

    # Multipliers for filtering
    product_multiplier = 1.0 if "all" in product_filter else 0.8
    channel_multiplier = 1.0 if "all" in channel_filter else 0.9
    if "social_media" in channel_filter and "all" not in channel_filter:
        channel_multiplier = 0.6

    # Health score data
    health_score_data = generate_health_score_data()
    current_health_data = health_score_data.get(selected_time, health_score_data["month"])

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
            xaxis=dict(showgrid=False, showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9)),
            yaxis=dict(showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(color='#4a4a4f', size=9), range=[min(current_health_data["values"]) - 2, max(current_health_data["values"]) + 2])
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
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

    # Customer Voice Snapshot
    st.markdown("## Customer Voice Snapshot")
    voice_view = st.radio("View", ["Overview", "Sentiment", "Intent", "Volume"], horizontal=True, key="voice_view")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Sentiment Distribution")
        sentiment_data = pd.DataFrame({
            'Category': ['Positive', 'Neutral', 'Negative'],
            'Value': [
                (60 + random.random() * 10) * product_multiplier,
                (20 + random.random() * 5) * product_multiplier,
                (10 + random.random() * 5) * product_multiplier
            ]
        })
        fig = px.pie(sentiment_data, values='Value', names='Category', color='Category', color_discrete_map={'Positive': '#34c759', 'Neutral': '#a2a2a7', 'Negative': '#ff3b30'}, hole=0.75)
        fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5, font=dict(size=10)), showlegend=True)
        fig.update_traces(textinfo='percent', textfont_size=10)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Intent Distribution")
        intent_data = pd.DataFrame({
            'Intent': ['Info Seeking', 'Complaint', 'Service Request', 'Feedback'],
            'Value': [
                35 + random.random() * 10,
                20 + random.random() * 5,
                20 + random.random() * 5,
                10 + random.random() * 5
            ]
        })
        fig = px.bar(intent_data, y='Intent', x='Value', orientation='h', color='Intent', color_discrete_map={'Info Seeking': '#007aff', 'Complaint': '#ff9500', 'Service Request': '#5856d6', 'Feedback': '#ffcc00'})
        fig.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True), yaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True), showlegend=False)
        fig.update_traces(marker_line_width=0, marker_line_color='rgba(0,0,0,0)', width=0.6)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("### Volume Trend (30 Days)")
        days = list(range(1, 31))
        volume_data = [(400 + random.random() * 300 + i * 5) * channel_multiplier for i in range(30)]
        vol_df = pd.DataFrame({'Day': days, 'Volume': volume_data})
        fig = px.line(vol_df, x='Day', y='Volume', line_shape='spline')
        fig.update_traces(line_color='#007aff', fill='tozeroy', fillcolor='rgba(0,122,255,0.18)', mode='lines')
        fig.update_layout(height=230, margin=dict(l=0, r=10, t=20, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(title=None, showgrid=False, showline=False, showticklabels=True, tickmode='array', tickvals=[1, 5, 10, 15, 20, 25, 30], tickfont=dict(size=9)), yaxis=dict(title=None, showgrid=True, gridcolor='#e5e5ea', showline=False, showticklabels=True, tickfont=dict(size=9), range=[min(volume_data) - 20, max(volume_data) + 20]))
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("Positive sentiment leads at 65%. Information-seeking is top intent (40%). Volume shows steady increase.")

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

    def generate_response(prompt):
        prompt_lower = prompt.lower()
        if "health score" in prompt_lower:
            return f"Customer Health Score is {current_health_data['score']}%, {current_health_data['trend']} {current_health_data['trend_label']}. Want to dive deeper?"
        elif "alerts" in prompt_lower:
            return "Current alerts include a spike in negative sentiment for mobile app update X.Y and high churn risk from billing errors. Need details or action suggestions?"
        elif "hotspots" in prompt_lower:
            return "Hotspots: Overdraft policy confusion (medium impact) and international transfer UI issues (low impact). Should we explore these further?"
        elif "opportunities" in prompt_lower:
            return "Top opportunities: Promote instant card activation, target mortgage inquiries, and scale service excellence. Which one interests you?"
        elif "themes" in prompt_lower:
            return "Positive themes: Fast customer service, easy mobile banking. Negative themes: App issues, wait times. Want to analyze a specific theme?"
        elif "thank" in prompt_lower:
            return "You're welcome! Anything else I can assist with?"
        else:
            return "I can help with insights on health scores, alerts, hotspots, opportunities, or themes. Try asking about one of these!"

    if prompt := st.chat_input("Ask about insights, alerts, or anything else..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            response = generate_response(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    st.markdown(f"## {page}")
    st.write("This section is under development. Please select 'Dashboard' from the sidebar to view the main dashboard.")
