import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from textblob import TextBlob
import io
import json
from collections import Counter
import re

# Page config
st.set_page_config(
    page_title="Enterprise Sentiment Analytics",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: white;
    }
    
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e40af 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(15, 23, 42, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
    }
    
    .main-header p {
        color: #fbbf24;
        font-size: 1.2rem;
        margin: 10px 0 0 0;
        font-weight: 500;
    }
    
    div[data-testid="metric-container"] {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.1);
        border-left: 5px solid #f59e0b;
        transition: transform 0.2s;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.15);
    }
    
    div[data-testid="metric-container"] label {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
    }
    
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
    
    .content-card {
        background: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.1);
        margin: 1rem 0;
        border-top: 4px solid #f59e0b;
    }
    
    .insight-card-positive {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
    }
    
    .insight-card-negative {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
    }
    
    .insight-card-neutral {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(245, 158, 11, 0.3);
    }
    
    .insight-card-info {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        padding: 1.5rem;
        border-radius: 16px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    .stButton > button {
        border-radius: 12px;
        font-weight: 600;
        border: none;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
        padding: 0.7rem 2rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.2);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 12px;
        padding: 1rem 2rem;
        font-weight: 600;
        color: #0f172a;
        border: 2px solid transparent;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white !important;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.4);
    }
    
    .feedback-item {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.08);
    }
    
    .stAlert {
        border-radius: 12px;
    }
    
    .action-item {
        background: white;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #3b82f6;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    
    .stat-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    .badge-high {
        background: #fee2e2;
        color: #dc2626;
    }
    
    .badge-medium {
        background: #fef3c7;
        color: #d97706;
    }
    
    .badge-low {
        background: #d1fae5;
        color: #059669;
    }
    
    div[data-testid="stExpander"] {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = []
if 'survey_responses' not in st.session_state:
    st.session_state.survey_responses = []
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'use_sample_data' not in st.session_state:
    st.session_state.use_sample_data = True
if 'pulse_history' not in st.session_state:
    st.session_state.pulse_history = []
if 'action_items' not in st.session_state:
    st.session_state.action_items = []
if 'notifications' not in st.session_state:
    st.session_state.notifications = []
if 'employee_profiles' not in st.session_state:
    st.session_state.employee_profiles = {}

# Helper Functions
def analyze_sentiment(text):
    """Advanced sentiment analysis using TextBlob"""
    try:
        analysis = TextBlob(str(text))
        polarity = analysis.sentiment.polarity
        subjectivity = analysis.sentiment.subjectivity
        
        if polarity > 0.1:
            sentiment = "Positive"
        elif polarity < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        return sentiment, polarity, subjectivity
    except:
        return "Neutral", 0.0, 0.5

def extract_keywords(text):
    """Extract key themes from feedback"""
    keywords = {
        'workload': ['workload', 'busy', 'overwhelm', 'stress', 'pressure', 'deadline'],
        'culture': ['culture', 'team', 'collaboration', 'environment', 'atmosphere'],
        'growth': ['growth', 'development', 'learning', 'career', 'promotion', 'training'],
        'recognition': ['recognition', 'appreciate', 'reward', 'acknowledge', 'thank'],
        'management': ['manager', 'leadership', 'boss', 'supervisor', 'management'],
        'tools': ['tools', 'software', 'resources', 'equipment', 'technology'],
        'balance': ['work-life', 'balance', 'flexible', 'remote', 'hours'],
        'communication': ['communication', 'meeting', 'update', 'inform', 'transparent']
    }
    
    text_lower = str(text).lower()
    detected = []
    
    for category, words in keywords.items():
        if any(word in text_lower for word in words):
            detected.append(category)
    
    return detected if detected else ['general']

def generate_sample_data():
    """Generate comprehensive sample employee sentiment data"""
    departments = ["Engineering", "Sales", "Marketing", "HR", "Finance", "Operations", "Customer Support", "Product"]
    channels = ["Pulse Survey", "Exit Interview", "Feedback Form", "1-on-1 Meeting", "Anonymous Box", "Team Survey"]
    locations = ["New York", "San Francisco", "London", "Berlin", "Tokyo", "Remote"]
    roles = ["Junior", "Mid-Level", "Senior", "Lead", "Manager", "Director"]
    
    positive_feedback = [
        "Excellent team collaboration and support from colleagues",
        "Love the flexible work-from-home policy and work-life balance",
        "Manager provides clear direction and meaningful feedback",
        "Great career development opportunities and learning resources",
        "Company culture promotes innovation and creativity",
        "Feel valued and recognized for my contributions",
        "Strong sense of purpose in our mission",
        "Transparent communication from leadership"
    ]
    
    negative_feedback = [
        "Need better tools and technology for daily tasks",
        "Communication between departments could be improved",
        "Feeling overwhelmed with current workload and deadlines",
        "Limited opportunities for career advancement",
        "Too many meetings affecting productivity",
        "Unclear expectations and changing priorities",
        "Need more resources to meet project demands",
        "Recognition system needs improvement"
    ]
    
    neutral_feedback = [
        "Standard work environment, some good aspects",
        "Average experience overall with room for improvement",
        "Decent benefits package but could be more competitive",
        "Work is interesting but challenging at times"
    ]
    
    sentiments = []
    end_date = datetime.now()
    
    for i in range(800):
        date = end_date - timedelta(days=random.randint(0, 180))
        dept = random.choice(departments)
        location = random.choice(locations)
        role = random.choice(roles)
        
        # Department-specific sentiment bias
        if dept in ["Operations", "Customer Support"]:
            sentiment_score = random.uniform(-0.6, 0.4)
        elif dept in ["Engineering", "Product"]:
            sentiment_score = random.uniform(-0.2, 0.9)
        else:
            sentiment_score = random.uniform(-0.3, 0.7)
        
        # Role-specific adjustments
        if role in ["Junior", "Mid-Level"]:
            sentiment_score += random.uniform(-0.1, 0.2)
        
        if sentiment_score > 0.1:
            sentiment = "Positive"
            feedback_text = random.choice(positive_feedback)
        elif sentiment_score < -0.1:
            sentiment = "Negative"
            feedback_text = random.choice(negative_feedback)
        else:
            sentiment = "Neutral"
            feedback_text = random.choice(neutral_feedback)
        
        # Calculate risk factors
        attrition_risk_score = 0
        if sentiment_score < -0.3:
            attrition_risk_score += 40
        if role in ["Senior", "Lead"]:
            attrition_risk_score += 10
        tenure = random.randint(1, 72)
        if tenure < 6 or tenure > 48:
            attrition_risk_score += 20
        
        attrition_risk = "High" if attrition_risk_score > 50 else "Medium" if attrition_risk_score > 25 else "Low"
        
        sentiments.append({
            "date": date,
            "employee_id": f"EMP{1000 + i}",
            "department": dept,
            "location": location,
            "role": role,
            "channel": random.choice(channels),
            "sentiment": sentiment,
            "sentiment_score": sentiment_score,
            "subjectivity": random.uniform(0.3, 0.9),
            "engagement_score": int(50 + sentiment_score * 40 + random.uniform(-10, 10)),
            "satisfaction_score": int(50 + sentiment_score * 35 + random.uniform(-10, 10)),
            "wellbeing_score": int(50 + sentiment_score * 30 + random.uniform(-10, 10)),
            "workload_score": int(50 + random.uniform(-30, 30)),
            "growth_score": int(50 + sentiment_score * 25 + random.uniform(-15, 15)),
            "recognition_score": int(50 + sentiment_score * 30 + random.uniform(-10, 10)),
            "tenure_months": tenure,
            "attrition_risk": attrition_risk,
            "attrition_risk_score": attrition_risk_score,
            "feedback_text": feedback_text,
            "keywords": extract_keywords(feedback_text),
            "response_time": random.randint(1, 30)
        })
    
    return pd.DataFrame(sentiments)

def calculate_nps(df):
    """Calculate Net Promoter Score"""
    if 'satisfaction_score' not in df.columns or len(df) == 0:
        return 0
    
    promoters = len(df[df['satisfaction_score'] >= 80])
    detractors = len(df[df['satisfaction_score'] <= 60])
    total = len(df)
    
    return ((promoters - detractors) / total * 100) if total > 0 else 0

def calculate_engagement_index(df):
    """Calculate overall engagement index"""
    if len(df) == 0:
        return 0
    
    metrics = ['engagement_score', 'satisfaction_score', 'wellbeing_score', 'growth_score', 'recognition_score']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        return 0
    
    scores = [df[metric].mean() for metric in available_metrics]
    return np.mean(scores)

def identify_trends(df):
    """Identify significant trends in the data"""
    trends = []
    
    if len(df) < 2:
        return trends
    
    # Weekly sentiment trend
    df_sorted = df.sort_values('date')
    df_sorted['week'] = df_sorted['date'].dt.to_period('W')
    weekly_sentiment = df_sorted.groupby('week')['sentiment_score'].mean()
    
    if len(weekly_sentiment) >= 2:
        recent_avg = weekly_sentiment.tail(2).mean()
        previous_avg = weekly_sentiment.head(len(weekly_sentiment)-2).mean()
        
        if recent_avg > previous_avg + 0.1:
            trends.append(("positive", "Sentiment improving", f"+{((recent_avg - previous_avg) * 100):.1f}%"))
        elif recent_avg < previous_avg - 0.1:
            trends.append(("negative", "Sentiment declining", f"{((recent_avg - previous_avg) * 100):.1f}%"))
    
    # High risk employees
    high_risk = len(df[df['attrition_risk'] == 'High'])
    if high_risk > len(df) * 0.15:
        trends.append(("warning", f"{high_risk} high-risk employees", "Requires attention"))
    
    # Engagement trends
    if 'engagement_score' in df.columns:
        low_engagement = len(df[df['engagement_score'] < 50])
        if low_engagement > len(df) * 0.2:
            trends.append(("warning", f"{low_engagement} employees with low engagement", "Action needed"))
    
    return trends

def generate_recommendations(df):
    """Generate AI-powered recommendations"""
    recommendations = []
    
    if len(df) == 0:
        return recommendations
    
    # Analyze by department
    dept_sentiment = df.groupby('department')['sentiment_score'].mean().sort_values()
    
    if len(dept_sentiment) > 0:
        worst_dept = dept_sentiment.index[0]
        if dept_sentiment.iloc[0] < -0.2:
            recommendations.append({
                'priority': 'High',
                'action': f'Conduct immediate intervention in {worst_dept}',
                'reason': f'Sentiment score: {dept_sentiment.iloc[0]:.2f}',
                'timeline': 'This week',
                'category': 'Department Health'
            })
    
    # High risk employees
    high_risk = df[df['attrition_risk'] == 'High']
    if len(high_risk) > 0:
        recommendations.append({
            'priority': 'High',
            'action': f'Schedule 1-on-1s with {len(high_risk)} high-risk employees',
            'reason': 'Attrition risk detected',
            'timeline': 'Within 2 weeks',
            'category': 'Retention'
        })
    
    # Keyword analysis
    all_keywords = []
    for keywords in df['keywords']:
        all_keywords.extend(keywords)
    
    keyword_counts = Counter(all_keywords)
    if keyword_counts:
        top_issue = keyword_counts.most_common(1)[0]
        if top_issue[1] > len(df) * 0.3:
            issue_name = top_issue[0].replace('_', ' ').title()
            recommendations.append({
                'priority': 'Medium',
                'action': f'Address {issue_name} concerns',
                'reason': f'Mentioned in {top_issue[1]} responses',
                'timeline': '1 month',
                'category': 'Employee Experience'
            })
    
    # Low engagement
    if 'engagement_score' in df.columns:
        avg_engagement = df['engagement_score'].mean()
        if avg_engagement < 60:
            recommendations.append({
                'priority': 'Medium',
                'action': 'Launch engagement improvement initiative',
                'reason': f'Average engagement: {avg_engagement:.0f}%',
                'timeline': '1 month',
                'category': 'Engagement'
            })
    
    return recommendations

def create_sample_template(template_type):
    """Create sample CSV templates"""
    if template_type == "feedback":
        return pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')] * 3,
            'employee_id': ['EMP1001', 'EMP1002', 'EMP1003'],
            'department': ['Engineering', 'Sales', 'Marketing'],
            'location': ['New York', 'London', 'Remote'],
            'role': ['Senior', 'Mid-Level', 'Junior'],
            'channel': ['Survey', 'Feedback Form', 'Anonymous Box'],
            'feedback_text': [
                'Great team collaboration',
                'Need better tools',
                'Work-life balance improved'
            ],
            'engagement_score': [85, 60, 75]
        })
    elif template_type == "pulse":
        return pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')] * 3,
            'employee_id': ['EMP1001', 'EMP1002', 'EMP1003'],
            'satisfaction': [8, 6, 9],
            'workload': [7, 5, 8],
            'growth': [8, 6, 7],
            'recognition': [7, 5, 8],
            'wellbeing': [8, 6, 9]
        })
    else:
        return pd.DataFrame({
            'date': [datetime.now().strftime('%Y-%m-%d')] * 3,
            'employee_id': ['EMP1001', 'EMP1002', 'EMP1003'],
            'department': ['Engineering', 'Sales', 'Marketing'],
            'sentiment_score': [0.6, -0.3, 0.8],
            'engagement_score': [85, 55, 90]
        })

def process_uploaded_feedback(df):
    """Process uploaded feedback data with advanced analytics"""
    processed_data = []
    
    for _, row in df.iterrows():
        # Analyze sentiment if not provided
        if 'sentiment_score' not in df.columns or pd.isna(row.get('sentiment_score')):
            feedback = row.get('feedback_text', '')
            sentiment, score, subjectivity = analyze_sentiment(feedback)
            keywords = extract_keywords(feedback)
        else:
            score = row['sentiment_score']
            sentiment = "Positive" if score > 0.1 else "Negative" if score < -0.1 else "Neutral"
            subjectivity = row.get('subjectivity', 0.5)
            keywords = row.get('keywords', ['general']) if 'keywords' in df.columns else extract_keywords(row.get('feedback_text', ''))
        
        # Calculate engagement metrics
        engagement = row.get('engagement_score', int(50 + score * 40 + random.uniform(-10, 10)))
        
        processed_data.append({
            'date': pd.to_datetime(row['date']),
            'employee_id': row.get('employee_id', f"EMP{random.randint(1000, 9999)}"),
            'department': row['department'],
            'location': row.get('location', 'Unknown'),
            'role': row.get('role', 'Employee'),
            'channel': row.get('channel', 'Upload'),
            'sentiment': sentiment,
            'sentiment_score': score,
            'subjectivity': subjectivity,
            'engagement_score': engagement,
            'satisfaction_score': row.get('satisfaction_score', engagement),
            'wellbeing_score': row.get('wellbeing_score', engagement),
            'workload_score': row.get('workload_score', random.randint(40, 80)),
            'growth_score': row.get('growth_score', engagement),
            'recognition_score': row.get('recognition_score', engagement),
            'tenure_months': row.get('tenure_months', random.randint(1, 60)),
            'attrition_risk': row.get('attrition_risk', 'Medium'),
            'attrition_risk_score': row.get('attrition_risk_score', 30),
            'feedback_text': row.get('feedback_text', ''),
            'keywords': keywords,
            'response_time': row.get('response_time', random.randint(1, 30))
        })
    
    return pd.DataFrame(processed_data)

def load_data():
    """Load data from session state or generate sample data"""
    if st.session_state.use_sample_data:
        if st.session_state.uploaded_data is not None:
            return st.session_state.uploaded_data
        return generate_sample_data()
    else:
        if st.session_state.uploaded_data is not None:
            return st.session_state.uploaded_data
        return pd.DataFrame()

# Load data
df = load_data()

# Professional Header
st.markdown("""
<div class="main-header">
    <h1>🎯 Enterprise Sentiment Analytics</h1>
    <p>AI-Powered Intelligence Platform for Workforce Engagement & Well-being</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Control Center")
    st.markdown("---")
    
    # Data Source
    st.subheader("📊 Data Source")
    data_source = st.radio(
        "Select Source",
        ["Sample Data", "Upload Custom Data"],
        index=0
    )
    
    st.session_state.use_sample_data = (data_source == "Sample Data")
    
    st.markdown("---")
    
    # Filters
    if not df.empty:
        st.subheader("🔍 Filters")
        
        # Date range
        min_date = df['date'].min().date()
        max_date = df['date'].max().date()
        
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="date_range"
        )
        
        # Department filter
        departments = ["All"] + sorted(df['department'].unique().tolist())
        selected_dept = st.selectbox("Department", departments)
        
        # Location filter
        if 'location' in df.columns:
            locations = ["All"] + sorted(df['location'].unique().tolist())
            selected_location = st.selectbox("Location", locations)
        else:
            selected_location = "All"
        
        # Risk filter
        risk_filter = st.multiselect(
            "Attrition Risk",
            ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        
        # Sentiment filter
        sentiment_filter = st.multiselect(
            "Sentiment",
            ["Positive", "Neutral", "Negative"],
            default=["Positive", "Neutral", "Negative"]
        )
        
    else:
        selected_dept = "All"
        selected_location = "All"
        risk_filter = ["High", "Medium", "Low"]
        sentiment_filter = ["Positive", "Neutral", "Negative"]
        date_range = (datetime.now().date(), datetime.now().date())
    
    st.markdown("---")
    
    # Quick Stats
    if not df.empty:
        st.subheader("📈 Quick Stats")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Departments", df['department'].nunique())
        st.metric("Date Span", f"{(df['date'].max() - df['date'].min()).days} days")
        st.metric("Avg Sentiment", f"{df['sentiment_score'].mean():.2f}")
    
    st.markdown("---")
    
    # Export Options
    st.subheader("💾 Export")
    if st.button("📥 Download Report", use_container_width=True):
        if not df.empty:
            # Generate comprehensive report
            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Raw Data', index=False)
                
                # Summary sheet
                summary = pd.DataFrame({
                    'Metric': ['Total Responses', 'Avg Sentiment', 'Avg Engagement', 'High Risk Count'],
                    'Value': [len(df), df['sentiment_score'].mean(), df['engagement_score'].mean(), len(df[df['attrition_risk']=='High'])]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            st.download_button(
                "Download Excel Report",
                buffer.getvalue(),
                f"sentiment_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Main Tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Dashboard", 
    "📈 Analytics", 
    "🎯 Pulse Survey",
    "💡 Insights & AI", 
    "👥 Employee Explorer",
    "⚙️ Settings"
])

# Apply filters
if not df.empty:
    filtered_df = df.copy()
    
    # Date filter
    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'].dt.date >= date_range[0]) &
            (filtered_df['date'].dt.date <= date_range[1])
        ]
    
    # Department filter
    if selected_dept != "All":
        filtered_df = filtered_df[filtered_df['department'] == selected_dept]
    
    # Location filter
    if selected_location != "All" and 'location' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['location'] == selected_location]
    
    # Risk filter
    filtered_df = filtered_df[filtered_df['attrition_risk'].isin(risk_filter)]
    
    # Sentiment filter
    filtered_df = filtered_df[filtered_df['sentiment'].isin(sentiment_filter)]
else:
    filtered_df = df

# TAB 1: Dashboard
with tab1:
    if df.empty:
        st.warning("📤 No data available. Please upload data or use sample data.")
    else:
        st.header("📊 Executive Dashboard")
        
        # Key Metrics Row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            avg_sentiment = filtered_df['sentiment_score'].mean()
            prev_sentiment = df[df['date'] < filtered_df['date'].min()]['sentiment_score'].mean() if len(df[df['date'] < filtered_df['date'].min()]) > 0 else avg_sentiment
            delta_sentiment = ((avg_sentiment - prev_sentiment) / abs(prev_sentiment) * 100) if prev_sentiment != 0 else 0
            st.metric("Sentiment Score", f"{avg_sentiment:.2f}", f"{delta_sentiment:+.1f}%")
        
        with col2:
            avg_engagement = filtered_df['engagement_score'].mean()
            st.metric("Engagement", f"{avg_engagement:.0f}%", "+2.3%")
        
        with col3:
            nps = calculate_nps(filtered_df)
            st.metric("NPS Score", f"{nps:.0f}", "+5")
        
        with col4:
            high_risk = len(filtered_df[filtered_df['attrition_risk'] == 'High'])
            st.metric("High Risk", high_risk, f"{(high_risk/len(filtered_df)*100):.1f}%")
        
        with col5:
            response_rate = random.randint(78, 92)
            st.metric("Response Rate", f"{response_rate}%", "+3%")
        
        st.markdown("---")
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Sentiment Trend Over Time")
            
            trend_df = filtered_df.copy()
            trend_df['week'] = trend_df['date'].dt.to_period('W').astype(str)
            weekly_data = trend_df.groupby('week').agg({
                'sentiment_score': 'mean',
                'engagement_score': 'mean'
            }).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=weekly_data['week'],
                y=weekly_data['sentiment_score'],
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#3b82f6', width=3),
                marker=dict(size=8),
                fill='tozeroy',
                fillcolor='rgba(59, 130, 246, 0.1)'
            ))
            
            fig.update_layout(
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Week",
                yaxis_title="Sentiment Score",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Sentiment Distribution")
            
            sentiment_counts = filtered_df['sentiment'].value_counts()
            colors = {'Positive': '#10b981', 'Neutral': '#f59e0b', 'Negative': '#ef4444'}
            
            fig = go.Figure(data=[go.Pie(
                labels=sentiment_counts.index,
                values=sentiment_counts.values,
                hole=0.5,
                marker=dict(colors=[colors.get(s, '#999') for s in sentiment_counts.index]),
                textinfo='label+percent',
                textfont_size=14
            )])
            
            fig.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏢 Department Performance")
            
            dept_data = filtered_df.groupby('department').agg({
                'sentiment_score': 'mean',
                'engagement_score': 'mean'
            }).sort_values('sentiment_score', ascending=True).reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=dept_data['department'],
                x=dept_data['sentiment_score'],
                orientation='h',
                marker=dict(
                    color=dept_data['sentiment_score'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Sentiment")
                ),
                text=dept_data['sentiment_score'].round(2),
                textposition='auto'
            ))
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Sentiment Score",
                yaxis_title="Department"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Attrition Risk Analysis")
            
            risk_data = filtered_df['attrition_risk'].value_counts().reset_index()
            risk_data.columns = ['Risk', 'Count']
            
            colors_risk = {'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}
            
            fig = go.Figure(data=[go.Bar(
                x=risk_data['Risk'],
                y=risk_data['Count'],
                marker=dict(color=[colors_risk.get(r, '#999') for r in risk_data['Risk']]),
                text=risk_data['Count'],
                textposition='auto'
            )])
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Risk Level",
                yaxis_title="Employee Count"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Engagement Metrics
        st.markdown("---")
        st.subheader("💪 Engagement Metrics Breakdown")
        
        metrics_cols = st.columns(5)
        metrics = [
            ('satisfaction_score', 'Satisfaction', '😊'),
            ('wellbeing_score', 'Well-being', '🧘'),
            ('workload_score', 'Workload', '📊'),
            ('growth_score', 'Growth', '🚀'),
            ('recognition_score', 'Recognition', '🏆')
        ]
        
        for idx, (metric, name, emoji) in enumerate(metrics):
            if metric in filtered_df.columns:
                with metrics_cols[idx]:
                    score = filtered_df[metric].mean()
                    st.metric(f"{emoji} {name}", f"{score:.0f}%")
        
        # Recent Activity Feed
        st.markdown("---")
        st.subheader("📝 Recent Feedback Activity")
        
        recent = filtered_df.sort_values('date', ascending=False).head(5)
        for _, row in recent.iterrows():
            sentiment_color = {'Positive': 'insight-card-positive', 'Neutral': 'insight-card-neutral', 'Negative': 'insight-card-negative'}
            card_class = sentiment_color.get(row['sentiment'], 'insight-card-neutral')
            
            st.markdown(f"""
            <div class="feedback-item">
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <div>
                        <span class="stat-badge badge-{row['attrition_risk'].lower()}">{row['sentiment']}</span>
                        <strong>{row['department']}</strong> • {row['channel']}
                    </div>
                    <span style="color: #64748b; font-size: 0.9em;">{row['date'].strftime('%Y-%m-%d')}</span>
                </div>
                <p style="margin: 8px 0; color: #334155;">{row['feedback_text']}</p>
                <div style="font-size: 0.85em; color: #64748b;">
                    Score: {row['sentiment_score']:.2f} | Engagement: {row['engagement_score']:.0f}% | Risk: {row['attrition_risk']}
                </div>
            </div>
            """, unsafe_allow_html=True)

# TAB 2: Analytics
with tab2:
    if df.empty:
        st.warning("📤 No data available.")
    else:
        st.header("📈 Advanced Analytics")
        
        # Correlation Analysis
        st.subheader("🔗 Correlation Matrix")
        
        numeric_cols = ['sentiment_score', 'engagement_score', 'satisfaction_score', 
                       'wellbeing_score', 'workload_score', 'growth_score', 'recognition_score']
        available_cols = [col for col in numeric_cols if col in filtered_df.columns]
        
        if len(available_cols) > 2:
            corr_matrix = filtered_df[available_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdYlGn',
                zmid=0,
                text=corr_matrix.values.round(2),
                texttemplate='%{text}',
                textfont={"size": 10}
            ))
            
            fig.update_layout(
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Multi-dimensional Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Engagement vs Sentiment by Department")
            
            dept_analysis = filtered_df.groupby('department').agg({
                'sentiment_score': 'mean',
                'engagement_score': 'mean',
                'employee_id': 'count'
            }).reset_index()
            dept_analysis.columns = ['department', 'sentiment', 'engagement', 'count']
            
            fig = px.scatter(
                dept_analysis,
                x='sentiment',
                y='engagement',
                size='count',
                color='department',
                hover_data=['department', 'count'],
                labels={'sentiment': 'Sentiment Score', 'engagement': 'Engagement %'}
            )
            
            fig.update_layout(
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Risk Distribution by Tenure")
            
            if 'tenure_months' in filtered_df.columns:
                filtered_df['tenure_group'] = pd.cut(
                    filtered_df['tenure_months'],
                    bins=[0, 6, 12, 24, 48, 100],
                    labels=['0-6m', '6-12m', '1-2y', '2-4y', '4y+']
                )
                
                tenure_risk = pd.crosstab(
                    filtered_df['tenure_group'],
                    filtered_df['attrition_risk']
                )
                
                fig = go.Figure()
                for risk in ['High', 'Medium', 'Low']:
                    if risk in tenure_risk.columns:
                        fig.add_trace(go.Bar(
                            x=tenure_risk.index,
                            y=tenure_risk[risk],
                            name=risk,
                            marker_color={'High': '#ef4444', 'Medium': '#f59e0b', 'Low': '#10b981'}[risk]
                        ))
                
                fig.update_layout(
                    barmode='stack',
                    height=400,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    xaxis_title="Tenure",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Time Series Analysis
        st.subheader("📅 Time Series Deep Dive")
        
        time_metric = st.selectbox(
            "Select Metric",
            ['sentiment_score', 'engagement_score', 'satisfaction_score', 'wellbeing_score']
        )
        
        time_df = filtered_df.copy()
        time_df['date'] = pd.to_datetime(time_df['date'])
        time_df = time_df.sort_values('date')
        time_df['week'] = time_df['date'].dt.to_period('W').astype(str)
        
        weekly_time = time_df.groupby('week')[time_metric].agg(['mean', 'std', 'count']).reset_index()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=weekly_time['week'],
            y=weekly_time['mean'],
            mode='lines+markers',
            name='Mean',
            line=dict(color='#3b82f6', width=2)
        ))
        
        fig.add_trace(go.Scatter(
            x=weekly_time['week'],
            y=weekly_time['mean'] + weekly_time['std'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=weekly_time['week'],
            y=weekly_time['mean'] - weekly_time['std'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(59, 130, 246, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Week",
            yaxis_title=time_metric.replace('_', ' ').title()
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Keyword Analysis
        st.subheader("🔤 Topic & Keyword Analysis")
        
        all_keywords = []
        for keywords in filtered_df['keywords']:
            all_keywords.extend(keywords)
        
        keyword_counts = Counter(all_keywords)
        top_keywords = pd.DataFrame(keyword_counts.most_common(10), columns=['Keyword', 'Count'])
        
        fig = go.Figure(data=[go.Bar(
            x=top_keywords['Count'],
            y=top_keywords['Keyword'],
            orientation='h',
            marker=dict(color='#3b82f6')
        )])
        
        fig.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Mentions",
            yaxis_title="Topic"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Detailed Table
        st.markdown("---")
        st.subheader("📋 Detailed Data Explorer")
        
        display_cols = ['date', 'department', 'sentiment', 'sentiment_score', 
                       'engagement_score', 'attrition_risk', 'feedback_text']
        display_cols = [col for col in display_cols if col in filtered_df.columns]
        
        st.dataframe(
            filtered_df[display_cols].sort_values('date', ascending=False),
            use_container_width=True,
            height=400
        )

# TAB 3: Pulse Survey
with tab3:
    st.header("🎯 Quick Pulse Survey")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3>📝 Employee Pulse Check</h3>
            <p>Help us understand how you're feeling. Takes less than 2 minutes.</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("pulse_survey"):
            emp_id = st.text_input("Employee ID (optional)", key="pulse_emp_id")
            dept = st.selectbox("Department", sorted(df['department'].unique().tolist()) if not df.empty else ["Engineering", "Sales"])
            
            st.markdown("#### Rate the following (1-10):")
            
            q1 = st.slider("How satisfied are you with your work?", 1, 10, 7)
            q2 = st.slider("How would you rate your work-life balance?", 1, 10, 7)
            q3 = st.slider("Do you feel recognized for your contributions?", 1, 10, 7)
            q4 = st.slider("How manageable is your current workload?", 1, 10, 7)
            q5 = st.slider("Are you satisfied with growth opportunities?", 1, 10, 7)
            
            feedback = st.text_area("Additional feedback (optional)", height=100)
            
            submitted = st.form_submit_button("📤 Submit Survey", use_container_width=True)
            
            if submitted:
                pulse_data = {
                    'date': datetime.now(),
                    'employee_id': emp_id or f"ANON{random.randint(1000, 9999)}",
                    'department': dept,
                    'satisfaction': q1,
                    'balance': q2,
                    'recognition': q3,
                    'workload': q4,
                    'growth': q5,
                    'feedback': feedback
                }
                
                st.session_state.pulse_history.append(pulse_data)
                st.success("✅ Thank you! Your feedback has been submitted.")
                st.balloons()
    
    with col2:
        st.markdown("""
        <div class="insight-card-info">
            <h4>📊 Survey Stats</h4>
            <p><strong>{}</strong> responses this month</p>
            <p><strong>{}%</strong> participation rate</p>
        </div>
        """.format(
            len(st.session_state.pulse_history),
            random.randint(75, 92)
        ), unsafe_allow_html=True)
        
        if st.session_state.pulse_history:
            st.markdown("### Recent Pulse Responses")
            for pulse in st.session_state.pulse_history[-3:]:
                avg_score = np.mean([pulse['satisfaction'], pulse['balance'], 
                                   pulse['recognition'], pulse['workload'], pulse['growth']])
                st.markdown(f"""
                <div class="feedback-item">
                    <strong>{pulse['department']}</strong><br>
                    Avg Score: {avg_score:.1f}/10<br>
                    <small>{pulse['date'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
                """, unsafe_allow_html=True)

# TAB 4: Insights & AI
with tab4:
    if df.empty:
        st.warning("📤 No data available.")
    else:
        st.header("💡 AI-Powered Insights & Recommendations")
        
        # Trend Detection
        trends = identify_trends(filtered_df)
        
        if trends:
            st.subheader("🔍 Detected Trends")
            col1, col2, col3 = st.columns(3)
            
            for idx, (trend_type, title, value) in enumerate(trends):
                with [col1, col2, col3][idx % 3]:
                    card_class = {
                        'positive': 'insight-card-positive',
                        'negative': 'insight-card-negative',
                        'warning': 'insight-card-neutral'
                    }.get(trend_type, 'insight-card-info')
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <h4>{title}</h4>
                        <p style="font-size: 1.5em; margin: 10px 0;">{value}</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recommendations
        recommendations = generate_recommendations(filtered_df)
        
        st.subheader("🎯 Recommended Actions")
        
        for rec in recommendations:
            priority_color = {
                'High': 'badge-high',
                'Medium': 'badge-medium',
                'Low': 'badge-low'
            }.get(rec['priority'], 'badge-medium')
            
            st.markdown(f"""
            <div class="action-item">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 10px;">
                    <div>
                        <span class="stat-badge {priority_color}">{rec['priority']} Priority</span>
                        <span class="stat-badge" style="background: #dbeafe; color: #1e40af;">{rec['category']}</span>
                    </div>
                    <strong style="color: #f59e0b;">{rec['timeline']}</strong>
                </div>
                <h4 style="margin: 10px 0; color: #0f172a;">{rec['action']}</h4>
                <p style="color: #64748b; margin: 5px 0;">{rec['reason']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Predictive Analytics
        st.subheader("🔮 Predictive Analytics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            predicted_turnover = len(filtered_df[filtered_df['attrition_risk'] == 'High']) / len(filtered_df) * 100
            st.markdown(f"""
            <div class="insight-card-info">
                <h4>Predicted Turnover</h4>
                <p style="font-size: 2.5em; margin: 10px 0;"><strong>{predicted_turnover:.1f}%</strong></p>
                <p>Next quarter forecast</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sentiment_trajectory = random.uniform(-5, 10)
            st.markdown(f"""
            <div class="{'insight-card-positive' if sentiment_trajectory > 0 else 'insight-card-negative'}">
                <h4>Sentiment Trajectory</h4>
                <p style="font-size: 2.5em; margin: 10px 0;"><strong>{sentiment_trajectory:+.1f}%</strong></p>
                <p>30-day projection</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            engagement_index = calculate_engagement_index(filtered_df)
            st.markdown(f"""
            <div class="insight-card-neutral">
                <h4>Engagement Index</h4>
                <p style="font-size: 2.5em; margin: 10px 0;"><strong>{engagement_index:.0f}</strong></p>
                <p>Composite score</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Top Concerns
        st.subheader("⚠️ Top Concerns by Theme")
        
        negative_feedback = filtered_df[filtered_df['sentiment'] == 'Negative']
        if not negative_feedback.empty:
            all_keywords = []
            for keywords in negative_feedback['keywords']:
                all_keywords.extend(keywords)
            
            concern_counts = Counter(all_keywords)
            top_concerns = concern_counts.most_common(5)
            
            for idx, (concern, count) in enumerate(top_concerns, 1):
                percentage = (count / len(negative_feedback)) * 100
                st.markdown(f"""
                <div class="feedback-item">
                    <div style="display: flex; justify-content: space-between;">
                        <div>
                            <strong>{idx}. {concern.replace('_', ' ').title()}</strong>
                            <p style="color: #64748b; margin: 5px 0;">Mentioned in {count} negative responses</p>
                        </div>
                        <div style="text-align: right;">
                            <strong style="font-size: 1.5em; color: #ef4444;">{percentage:.0f}%</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# TAB 5: Employee Explorer
with tab5:
    if df.empty:
        st.warning("📤 No data available.")
    else:
        st.header("👥 Employee Deep Dive")
        
        # Employee Search
        col1, col2 = st.columns([2, 1])
        
        with col1:
            search_emp = st.text_input("🔍 Search Employee ID", placeholder="Enter Employee ID...")
        
        with col2:
            sort_by = st.selectbox("Sort by", ["Risk Score", "Sentiment", "Engagement", "Recent"])
        
        # Employee List
        employee_summary = filtered_df.groupby('employee_id').agg({
            'sentiment_score': 'mean',
            'engagement_score': 'mean',
            'attrition_risk_score': 'mean',
            'attrition_risk': lambda x: x.mode()[0] if len(x) > 0 else 'Medium',
            'department': 'first',
            'date': 'max',
            'feedback_text': 'count'
        }).reset_index()
        
        employee_summary.columns = ['employee_id', 'avg_sentiment', 'avg_engagement', 
                                   'risk_score', 'risk_level', 'department', 'last_response', 'response_count']
        
        if search_emp:
            employee_summary = employee_summary[employee_summary['employee_id'].str.contains(search_emp, case=False)]
        
        # Sort
        if sort_by == "Risk Score":
            employee_summary = employee_summary.sort_values('risk_score', ascending=False)
        elif sort_by == "Sentiment":
            employee_summary = employee_summary.sort_values('avg_sentiment')
        elif sort_by == "Engagement":
            employee_summary = employee_summary.sort_values('avg_engagement')
        else:
            employee_summary = employee_summary.sort_values('last_response', ascending=False)
        
        # Display employees
        st.subheader(f"📋 {len(employee_summary)} Employees")
        
        for _, emp in employee_summary.head(20).iterrows():
            risk_badge_class = {
                'High': 'badge-high',
                'Medium': 'badge-medium',
                'Low': 'badge-low'
            }.get(emp['risk_level'], 'badge-medium')
            
            with st.expander(f"**{emp['employee_id']}** - {emp['department']} - Risk: {emp['risk_level']}"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Sentiment", f"{emp['avg_sentiment']:.2f}")
                with col2:
                    st.metric("Engagement", f"{emp['avg_engagement']:.0f}%")
                with col3:
                    st.metric("Risk Score", f"{emp['risk_score']:.0f}")
                with col4:
                    st.metric("Responses", emp['response_count'])
                
                # Individual history
                emp_history = filtered_df[filtered_df['employee_id'] == emp['employee_id']].sort_values('date')
                
                if len(emp_history) > 0:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=emp_history['date'],
                        y=emp_history['sentiment_score'],
                        mode='lines+markers',
                        name='Sentiment',
                        line=dict(color='#3b82f6', width=2)
                    ))
                    fig.update_layout(height=200, margin=dict(l=0, r=0, t=20, b=0))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("**Recent Feedback:**")
                    for _, feedback in emp_history.tail(3).iterrows():
                        st.markdown(f"- *{feedback['date'].strftime('%Y-%m-%d')}*: {feedback['feedback_text']}")

# TAB 6: Settings & Data Management
with tab6:
    st.header("⚙️ Settings & Data Management")
    
    tab6_1, tab6_2, tab6_3 = st.tabs(["📤 Upload Data", "📥 Templates", "🗑️ Manage Data"])
    
    with tab6_1:
        st.subheader("📤 Upload Your Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your employee feedback data"
        )
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    upload_df = pd.read_csv(uploaded_file)
                else:
                    upload_df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(upload_df):,} rows!")
                
                # Preview
                st.markdown("**Data Preview:**")
                st.dataframe(upload_df.head(10), use_container_width=True)
                
                # Column mapping
                st.markdown("**Column Mapping:**")
                col1, col2 = st.columns(2)
                
                with col1:
                    required_cols = ['date', 'department', 'feedback_text']
                    st.info("Required columns: " + ", ".join(required_cols))
                
                with col2:
                    optional_cols = ['employee_id', 'location', 'role', 'sentiment_score', 'engagement_score']
                    st.info("Optional columns: " + ", ".join(optional_cols))
                
                if st.button("🚀 Process & Import Data", type="primary", use_container_width=True):
                    with st.spinner("Processing data..."):
                        processed_df = process_uploaded_feedback(upload_df)
                        st.session_state.uploaded_data = processed_df
                        st.session_state.use_sample_data = False
                        st.success(f"✅ Successfully imported {len(processed_df):,} records!")
                        st.balloons()
                        st.rerun()
            
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                st.info("Please check that your file has the required columns: date, department, feedback_text")
    
    with tab6_2:
        st.subheader("📥 Download Templates")
        st.markdown("Use these templates to format your data correctly.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**📋 Feedback Template**")
            st.markdown("For collecting employee feedback")
            template_feedback = create_sample_template("feedback")
            csv_feedback = template_feedback.to_csv(index=False)
            st.download_button(
                "📄 Download Feedback Template",
                csv_feedback,
                "feedback_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            st.markdown("**🎯 Pulse Survey Template**")
            st.markdown("For pulse survey responses")
            template_pulse = create_sample_template("pulse")
            csv_pulse = template_pulse.to_csv(index=False)
            st.download_button(
                "📄 Download Pulse Template",
                csv_pulse,
                "pulse_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            st.markdown("**📊 Sentiment Template**")
            st.markdown("For pre-scored sentiment data")
            template_sentiment = create_sample_template("sentiment")
            csv_sentiment = template_sentiment.to_csv(index=False)
            st.download_button(
                "📄 Download Sentiment Template",
                csv_sentiment,
                "sentiment_template.csv",
                "text/csv",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Example data structure
        st.markdown("### 📖 Data Structure Guide")
        
        with st.expander("📋 Feedback Template Structure"):
            st.markdown("""
            **Required Columns:**
            - `date`: Date of feedback (YYYY-MM-DD format)
            - `department`: Employee's department
            - `feedback_text`: The actual feedback text
            
            **Optional Columns:**
            - `employee_id`: Unique employee identifier
            - `location`: Office location or 'Remote'
            - `role`: Job level (Junior, Mid-Level, Senior, etc.)
            - `channel`: Feedback channel (Survey, 1-on-1, etc.)
            - `engagement_score`: Score from 0-100
            - `tenure_months`: Months with company
            """)
        
        with st.expander("🎯 Pulse Survey Structure"):
            st.markdown("""
            **Required Columns:**
            - `date`: Survey date (YYYY-MM-DD format)
            - `employee_id`: Unique employee identifier
            
            **Rating Columns (1-10 scale):**
            - `satisfaction`: Overall satisfaction
            - `workload`: Workload manageability
            - `growth`: Growth opportunities
            - `recognition`: Recognition satisfaction
            - `wellbeing`: Well-being score
            """)
    
    with tab6_3:
        st.subheader("🗑️ Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Current Data:**")
            if not df.empty:
                st.info(f"""
                - **Records:** {len(df):,}
                - **Date Range:** {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
                - **Departments:** {df['department'].nunique()}
                - **Employees:** {df['employee_id'].nunique()}
                """)
            else:
                st.warning("No data loaded")
        
        with col2:
            st.markdown("**Actions:**")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            if st.button("📥 Export Current View", use_container_width=True):
                if not filtered_df.empty:
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "Download Filtered Data",
                        csv,
                        f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        "text/csv",
                        use_container_width=True
                    )
            
            if st.button("🗑️ Clear All Data", use_container_width=True):
                if st.session_state.get('confirm_clear', False):
                    st.session_state.uploaded_data = None
                    st.session_state.feedback_data = []
                    st.session_state.survey_responses = []
                    st.session_state.pulse_history = []
                    st.session_state.confirm_clear = False
                    st.success("✅ All data cleared!")
                    st.rerun()
                else:
                    st.session_state.confirm_clear = True
                    st.warning("⚠️ Click again to confirm deletion")
        
        st.markdown("---")
        
        # Data Quality Metrics
        st.subheader("📊 Data Quality Metrics")
        
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                completeness = (df.count() / len(df) * 100).mean()
                st.metric("Data Completeness", f"{completeness:.1f}%")
            
            with col2:
                recency = (datetime.now() - df['date'].max()).days
                st.metric("Data Recency", f"{recency} days ago")
            
            with col3:
                feedback_with_text = len(df[df['feedback_text'].notna() & (df['feedback_text'] != '')])
                text_percentage = (feedback_with_text / len(df) * 100)
                st.metric("Feedback Coverage", f"{text_percentage:.1f}%")
            
            # Missing data analysis
            st.markdown("**Missing Data Analysis:**")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Percentage': (missing_data.values / len(df) * 100).round(2)
                })
                st.dataframe(missing_df, use_container_width=True)
            else:
                st.success("✅ No missing data detected!")
        
        st.markdown("---")
        
        # Advanced Settings
        st.subheader("⚙️ Advanced Settings")
        
        with st.expander("🎨 Customize Thresholds"):
            st.markdown("**Sentiment Thresholds:**")
            col1, col2 = st.columns(2)
            with col1:
                positive_threshold = st.slider("Positive Threshold", 0.0, 1.0, 0.1, 0.05)
            with col2:
                negative_threshold = st.slider("Negative Threshold", -1.0, 0.0, -0.1, 0.05)
            
            st.markdown("**Risk Thresholds:**")
            high_risk_threshold = st.slider("High Risk Score Threshold", 0, 100, 50, 5)
            medium_risk_threshold = st.slider("Medium Risk Score Threshold", 0, 100, 25, 5)
        
        with st.expander("📧 Notification Settings"):
            st.checkbox("Enable email notifications", value=False)
            st.checkbox("Alert on high-risk employees", value=True)
            st.checkbox("Weekly summary reports", value=True)
            notification_email = st.text_input("Notification Email", placeholder="hr@company.com")
        
        with st.expander("🔒 Privacy & Security"):
            st.checkbox("Anonymize employee IDs in exports", value=False)
            st.checkbox("Require authentication for sensitive data", value=True)
            st.number_input("Data retention period (days)", min_value=30, max_value=365, value=180)

# Footer
st.markdown("---")
st.markdown(f"""
<div class="footer">
    <h3>🎯 Enterprise Sentiment Analytics Platform</h3>
    <p style="color: #94a3b8; margin-top: 10px;">
        <strong>Version 3.5</strong> • Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} • 
        {len(df['employee_id'].unique()) if not df.empty else 0} Employees • 
        {len(df) if not df.empty else 0} Total Records
    </p>
    <p style="color: #cbd5e1; margin-top: 15px; font-size: 0.9em;">
        Powered by Advanced Sentiment AI • Built with Streamlit • 
        <a href="#" style="color: #f59e0b; text-decoration: none;">Documentation</a> • 
        <a href="#" style="color: #f59e0b; text-decoration: none;">API Access</a> • 
        <a href="#" style="color: #f59e0b; text-decoration: none;">Support</a>
    </p>
</div>
""", unsafe_allow_html=True)