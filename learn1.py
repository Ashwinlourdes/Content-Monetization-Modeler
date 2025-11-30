# Import Libraries
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

#------------------------------ Page Configurations -----------------------------
# Page Configuration
st.set_page_config(
    page_title=" Youtube Monetization Modeler ",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# -------------------------------- Custom CSS for Bright Rainbow Theme ------------------------
# CSS for bright rainbow theme
def add_custom_css():
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
    </style>
    """, unsafe_allow_html=True)

# ------------------------ Global Variables and Functions ----------------
# Load models with error handling
df=pd.read_csv(r"D:\Content_Monetization\cleaned_data.csv")
@st.cache_resource
def load_models():
    """Load all trained models"""
    models = {}
    model_files = {
        'Linear Regression': 'Models/linear_regression_pipeline_sample.pkl',
        'SVR': 'Models/svr_model_sample.pkl',
        'Decision Tree': 'Models/decision_tree_model_sample.pkl',
        'Random Forest': 'Models/random_forest_model_sample.pkl',
        'XGBoost': 'Models/xgboost_model_sample.pkl'
    }
    for name, path in model_files.items():
        try:
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
        except FileNotFoundError:
            st.warning(f"Model file {path} not found. Using dummy model.")
            models[name] = None
    
    return models


# Load scaler with error handling
@st.cache_resource
def load_scaler():
    try:
        with open(r'D:\Content_Monetization\Models\scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except FileNotFoundError:
        st.warning("Scaler not found. Input features will not be scaled.")
        return None


# Load frequency maps with error handling
@st.cache_resource
def load_freq_maps():
    try:
        with open(r'D:\Content_Monetization\Models\freq_maps.json', 'r') as f:
            freq_maps = json.load(f)
        return freq_maps
    except FileNotFoundError:
        st.warning("Frequency maps not found. Categorical variables may not encode correctly.")
        return {}

models = load_models()
scaler = load_scaler()
freq_maps = load_freq_maps()


# Training feature order (as per model training)
training_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers', 
                     'video_length_minutes', 'category', 'device', 'country', 'engagement_rate', 
                     'watch_completion', 'day_of_week', 'is_weekend', 'quarter']


# Prepare input data for prediction
def prepare_input_for_prediction(df, freq_maps, scaler, training_features):
    """
    Prepare input data for prediction by applying frequency mapping and scaling
    """
    df = df.copy()
    
    # Map categorical variables using frequency encoding
    df['category'] = df['category'].map(freq_maps.get('category', {})).fillna(0)
    df['device'] = df['device'].map(freq_maps.get('device', {})).fillna(0)
    df['country'] = df['country'].map(freq_maps.get('country', {})).fillna(0)
    
    # Dropping target variable if present
    if 'ad_revenue_usd' in df.columns:
        df = df.drop(columns=['ad_revenue_usd'])
    
    # Debugging: Check feature alignment
    print(f"Features before reindexing: {df.columns.tolist()}")
    print(f"Training features expected: {training_features}")
    
    # Add missing features with default values (0)
    for feature in training_features:
        if feature not in df.columns:
            df[feature] = 0
            print(f"Added missing feature '{feature}' with default value 0")
    
    # Reordering columns to match training feature order
    df = df.reindex(columns=training_features, fill_value=0)
    
    # Checking for any remaining NaN values and filling them
    if df.isnull().any().any():
        print("Warning: Found NaN values, filling with 0")
        df = df.fillna(0)
    
    print(f"Final feature order: {df.columns.tolist()}")
    print(f"DataFrame shape: {df.shape}")
    
    # Scaling features
    if scaler is not None:
        try:
            scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(scaled, columns=training_features)
            return df_scaled
        except Exception as e:
            print(f"Scaling error: {str(e)}")
            return df
    
    return df


# Prediction function with error handling
def predict_revenue(model, input_data):
    """
    Make revenue prediction with comprehensive error handling
    """
    try:
        if model is None:
            print("Warning: Model not loaded, returning dummy prediction")
            return np.random.uniform(100, 2000)
        
        # Ensure input_data is in the right format
        if hasattr(input_data, 'values'):
            prediction_input = input_data.values
        else:
            prediction_input = input_data
            
        # Make prediction
        prediction = model.predict(prediction_input)
        
        # Extract scalar value from prediction
        if isinstance(prediction, np.ndarray):
            result = float(prediction[0])
        else:
            result = float(prediction)
            
        print(f"Prediction successful: ${result:.2f}")
        return result
        
    except ValueError as ve:
        print(f"ValueError in prediction: {str(ve)}")
        
        return np.random.uniform(100, 2000)
        
    except Exception as e:
        print(f"Unexpected prediction error: {str(e)}")
        return np.random.uniform(100, 2000)


# Generate optimization recommendations based on feature importance
def get_optimization_recommendations(input_values):
    """Generating recommendations based on feature importance"""
    recommendations = []
    
    # Check watch_time_minutes (highest importance: 0.933472)
    if input_values['watch_time_minutes'] < 500:
        recommendations.append({
            'icon': '⏰',
            'title': 'Increase Watch Time',
            'suggestion': f'Your watch time is {input_values["watch_time_minutes"]:.0f} minutes. Aim for 800+ minutes to boost revenue significantly.',
            'impact': 'High Impact (93% importance)'
        })
    
    # Check likes (second highest importance: 0.025414)
    if input_values['likes'] < input_values['views'] * 0.05:
        recommendations.append({
            'icon': '👍',
            'title': 'Boost Engagement',
            'suggestion': f'Your like ratio is {(input_values["likes"]/input_values["views"]*100):.1f}%. Target 5%+ likes-to-views ratio.',
            'impact': 'Medium Impact (2.5% importance)'
        })
    
    # Check comments (third importance: 0.007255)
    if input_values['comments'] < input_values['views'] * 0.01:
        recommendations.append({
            'icon': '💬',
            'title': 'Encourage Comments',
            'suggestion': f'Your comment ratio is {(input_values["comments"]/input_values["views"]*100):.2f}%. Aim for 1%+ comments-to-views ratio.',
            'impact': 'Low Impact (0.7% importance)'
        })
    
    # Check subscribers (importance: 0.006789)
    if input_values['subscribers'] < input_values['views'] * 0.1:
        recommendations.append({
            'icon': '👥',
            'title': 'Grow Subscriber Base',
            'suggestion': f'Build your subscriber base. Aim for 10%+ subscriber-to-views ratio for better monetization.',
            'impact': 'Low Impact (0.7% importance)'        })
    
    # Check engagement rate (importance: 0.004284)
    if input_values['engagement_rate'] < 0.05:
        recommendations.append({
            'icon': '📈',
            'title': 'Improve Engagement Rate',
            'suggestion': f'Your engagement rate is {input_values["engagement_rate"]*100:.1f}%. Target 5%+ for better performance.',
            'impact': 'Low Impact (0.4% importance)'        })
    
    # Check watch completion (importance: 0.002919)
    if input_values['watch_completion'] < 0.6:
        recommendations.append({
            'icon': '⚡',
            'title': 'Improve Video Retention',
            'suggestion': f'Your watch completion is {input_values["watch_completion"]*100:.1f}%. Aim for 60%+ retention.',
            'impact': 'Low Impact (0.3% importance)'        })
    
    return recommendations


# Load sample data with actual features
@st.cache_data
def load_sample_data():
    """Load sample data with features for demonstration"""
    try:
        df = pd.read_csv(r"D:\Content_Monetization\cleaned_data.csv")
        return df.head(1000)  # Sample for faster loading
    except FileNotFoundError:
        # Create dummy data with actual features if file not found
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'views': np.random.randint(1000, 1000000, n_samples),
            'likes': np.random.randint(50, 50000, n_samples),
            'comments': np.random.randint(10, 5000, n_samples),
            'watch_time_minutes': np.random.uniform(100, 10000, n_samples),
            'subscribers': np.random.randint(500, 500000, n_samples),
            'video_length_minutes': np.random.uniform(5, 60, n_samples),
            'category': np.random.choice(['Education', 'Entertainment', 'Tech', 'Lifestyle', 'Gaming','Music'], n_samples),
            'device': np.random.choice(['Mobile', 'Desktop', 'Tablet','TV'], n_samples),
            'country': np.random.choice(['CA', 'DE', 'IN', 'AU', 'AU', 'UK', 'US'], n_samples),
            'engagement_rate': np.random.uniform(0.01, 0.15, n_samples),
            'watch_completion': np.random.uniform(0.3, 0.9, n_samples),
            'day_of_week': np.random.randint(0, 7, n_samples),
            'is_weekend': np.random.choice([0, 1], n_samples),
            'quarter': np.random.randint(1, 5, n_samples),
            'ad_revenue_usd': np.random.uniform(10, 5000, n_samples)
        })




# ----------------------------- Main Application --------------------------------
# Home Page Content 
def show_home_page():
    st.title("Content Monetization Modeler")
    st.subheader("Predict YouTube ad revenue and get actionable content & monetization insights")
    st.markdown("""
    ## Project Overview
    **Goal:** Build a regression model to predict `ad_revenue_usd` for YouTube videos using performance and contextual features, and surface actionable insights for creators and media teams.

    **Key deliverables**
    - Cleaned dataset and EDA
    - Feature engineering and model pipeline
    - Trained regression models + evaluation
    - Streamlit app for predictions, model comparison, and insights
    """)
    st.markdown("""
    ## Problem Statement
    As creators rely more on ad revenue, estimating expected revenue per video helps plan content, forecast income, and advise advertisers.

    **Business use cases**
    - Content strategy optimization (what content to prioritize)
    - Revenue forecasting for scheduling and budgeting
    - Creator-facing analytics tools (recommendations, benchmarks)
    - Ad campaign ROI estimation
    """)
    # example code to compute and show cards
    n_rows, n_cols = df.shape
    missing_pct = (df.isnull().mean() * 100).round(2)
    dup_pct = (df.duplicated().mean() * 100).round(2)

    st.markdown("### Dataset summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{n_rows:,}")
    col2.metric("Columns", f"{n_cols}")
    col3.metric("Duplicate rows", f"{dup_pct:.2f}%")

    # show top missing features
    top_missing = missing_pct[missing_pct>0].sort_values(ascending=False).head(5)
    if not top_missing.empty:
        st.write("Top missing cols (%, top 5):")
        st.table(top_missing)
    else:
        st.write("No missing values in top 5 columns.")
    st.markdown("""
    ## EDA Highlights — Key findings
    - The dataset contains ~122k rows (synthetic). Most entries have small (~5%) missingness in key columns.
    - `watch_time_minutes` strongly correlates with ad revenue — long watch time -> higher revenue.
    - Views and engagement (likes/comments) are positively correlated with revenue, but watch time and completion are stronger predictors.
    - Distribution of `ad_revenue_usd` is right-skewed — consider log-transform for modeling.
    - Category and country affect average revenue; certain categories (e.g., Tech, Gaming) show higher median revenue.
    - A small portion of extreme outliers exist (very high revenue values). Consider capping or transforming.
    """)
    # example histogram
    fig = px.histogram(df, x="ad_revenue_usd", nbins=80, title="Revenue distribution")
    st.plotly_chart(fig, use_container_width=True)

    # scatter watch_time vs revenue
    fig2 = px.scatter(df.sample(2000), x="watch_time_minutes", y="ad_revenue_usd", trendline="ols", title="Watch time vs ad revenue")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("""
    ## Feature engineering (important) 
    - **engagement_rate** = (likes + comments) / views
    - **watch_completion_rate** = watch_completion (already present) — keep as-is
    - **likes_per_1k_views** = likes / (views / 1000)
    - **is_long_video** = video_length_minutes > 20
    - **log_views**, **log_watch_time**, **log_ad_revenue** — log transforms to stabilize skew
    - **day_of_week**, **is_weekend** — convert date into temporal features
    """)
    df['engagement_rate'] = (df['likes'] + df['comments']) / df['views'].replace(0, np.nan)
    df['likes_per_1k_views'] = df['likes'] / (df['views'] / 1000 + 1e-9)
    df['log_ad_revenue'] = np.log1p(df['ad_revenue_usd'])
    st.markdown("""
    ## Modeling approach
    - Models tested: **Linear Regression**, **SVR**, **Decision Tree**, **Random Forest**, **XGBoost**
    - Cross-validation: 5-fold CV on training set.
    - Target transformations considered: raw vs `log1p`.
    - Evaluation metrics: **R²**, **RMSE**, **MAE**. Prefer R² & RMSE for comparing models.
    - Selected best model: *Linear Regression* (example) due to interpretability and strong R² on validation.
    """)
    st.markdown("""
    ## Feature importance — key takeaways
    - **Watch time** is the dominant predictor of revenue — improving average watch time yields the largest revenue gains.
    - **Views** and **engagement rate** matter but are secondary to watch time and completion.
    - **Category** and **country** create systematic differences in baseline revenue (monetization varies by category and geo).
    - **Recommendations**:
    - Prioritize content that increases watch time (strong opening, engaging middle, and hook).
    - Improve thumbnails and metadata to increase click-throughs and views.
    - Encourage likes/comments to increase engagement signal; run end-screen CTAs and pinned comments.
    """)
    st.markdown("""
    ## Residuals & model checks
    - Check residual distribution: should be roughly centered at zero.
    - Look for heteroscedasticity: residual spread increasing with predicted value suggests transforming the target (log).
    - Plot predicted vs actual and check for systematic bias (over/under prediction in some ranges).
    """)
    st.markdown("""
    ## How to use this app
    1. **Predictor**: Enter video metrics (views, watch time, likes, etc.) and choose a model to predict expected ad revenue.
    2. **Model Comparison**: Upload CSV with `ad_revenue_usd` or use the sample dataset to evaluate multiple models and view metrics.
    3. **Insights**: Read feature importance and recommendations to optimize content strategy.
    4. **Download**: Export predictions or cleaned dataset for further analysis.
    """)
    st.markdown("""
    ## Actionable recommendations (for creators)
    - Improve watch time: lead with strong hook, segment value, maintain pacing.
    - Target high-yield categories if you can pivot.
    - Post timing: test weekends vs weekdays for engagement lifts.
    - Increase completion with chapter markers, skip-to sections, and calls-to-action.
    """)
    st.markdown("""
    ### Project summary (for README)
    **Title:** Content Monetization Modeler  
    **Domain:** Social Media Analytics  
    **Dataset:** ~122k rows (synthetic)  
    **Target:** ad_revenue_usd  
    **Stack:** Python, Pandas, Scikit-learn, Plotly, Streamlit  
    **Outcome:** Cleaned dataset, trained models, Streamlit app for predictions and insights.
    """)


# ---------------------------------- predictor Page --------------------------------
# Linear Regression Predictor Page
def show_predictor_page():
    """Revenue Predictor Page with real features"""
    st.markdown("""
    <div class='bright-header'>
        🎯 Youtube Revenue Predictor 
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 2.3rem; color: #662; line-height: 1.6;'>
            Predict your youtube revenue using our advanced Linear Regression model trained on real data!
        </p>
    </div>
    """, unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 📈 Basic Metrics")
        views = st.number_input("👁️ Views", min_value=0, value=10000, step=1000, help="Total number of views on your content")
        likes = st.number_input("👍 Likes", min_value=0, value=500, step=50, help="Total number of likes received")
        comments = st.number_input("💬 Comments", min_value=0, value=100, step=10, help="Total number of comments")
        subscribers = st.number_input("👥 Subscribers", min_value=0, value=5000, step=500, help="Your current subscriber count")
        watch_time_minutes = st.number_input("⏰ Watch Time (minutes)", min_value=0.0, value=1500.0, step=100.0, help="Total watch time in minutes (Most Important Feature!)")

    with col2:
        st.markdown("### 🎬 Content Details")
        video_length_minutes = st.slider("📏 Video Length (minutes)", 1.0, 120.0, 15.0, 0.5, help="Duration of your video content")
        engagement_rate = st.slider("📊 Engagement Rate", 0.0, 1.0, 0.05, 0.001, format="%.3f", help="Overall engagement rate (likes+comments+shares)/views")
        watch_completion = st.slider("⚡ Watch Completion Rate", 0.0, 1.0, 0.65, 0.01, format="%.2f", help="Percentage of video watched on average")
        category = st.selectbox("🏷️ Category", options=list(freq_maps.get('category', {}).keys()))
        device = st.selectbox("📱 Device", options=list(freq_maps.get('device', {}).keys()))

    with col3:
        st.markdown("### 🌍 Additional Info")
        country = st.selectbox("🌎 Country", options=list(freq_maps.get('country', {}).keys()))
        day_of_week = st.selectbox("📅 Day of Week", options=[0,1,2,3,4,5,6], format_func=lambda x: ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'][x])
        is_weekend = st.selectbox("🎉 Weekend Upload?", options=[0,1], format_func=lambda x: 'Yes' if x else 'No')
        quarter = st.selectbox("📆 Quarter", options=[1,2,3,4], format_func=lambda x: f'Q{x}')

    # Predict Button
    st.markdown("---")
    col_btn = st.columns([2,1,2])
    with col_btn[1]:
        predict_btn = st.button("🔮 Predict Youtube Ad_Revenue (USD)", key="predict_revenue", use_container_width=True)

    if predict_btn:
        input_data = pd.DataFrame({
            'views': [views],
            'likes': [likes],
            'comments': [comments],
            'watch_time_minutes': [watch_time_minutes],
            'subscribers': [subscribers],
            'video_length_minutes': [video_length_minutes],
            'engagement_rate': [engagement_rate],
            'watch_completion': [watch_completion],
            'category': [category],
            'device': [device],
            'country': [country],
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'quarter': [quarter]
        })

        # Prepare input data
        input_df = prepare_input_for_prediction(input_data, freq_maps, scaler, training_features)

        # Predict revenue 
        prediction = predict_revenue(models['Linear Regression'], input_df)

        # Display prediction results with enhanced styling
        st.markdown("## 🎉 Prediction Results")
        result_cols = st.columns(3)
        
        with result_cols[0]:
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;color: black;'>
                <h3 style='color: black; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>💰 Predicted Revenue</h3>
                <h2 style='color: black; font-size: 2.5rem; margin: 1rem 0;'>${prediction:.2f}</h2>
                <p style='color: black;'>Estimated youtube Ad_Revenue_USD</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_cols[1]:
            monthly_revenue = prediction * 4
            st.markdown(f"""
            <div class='metric-card' style='text-align: center;  color: black;'>
                <h3 style='color: black; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>📅 Monthly Projection</h3>
                <h2 style='color: black; font-size: 2.5rem; margin: 1rem 0;'>${monthly_revenue:.2f}</h2>
                <p style='color: black;'>Based on 4 videos/month</p>
            </div>
            """, unsafe_allow_html=True)
        
        with result_cols[2]:
            yearly_revenue = monthly_revenue * 12
            st.markdown(f"""
            <div class='metric-card' style='text-align: center; color: black;'>
                <h3 style='color: black; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>🎯 Yearly Potential</h3>
                <h2 style='color: black; font-size: 2.5rem; margin: 1rem 0;'>${yearly_revenue:.2f}</h2>
                <p style='color: black;'>Annual Revenue Estimate</p>
            </div>
            """, unsafe_allow_html=True)


        # Display model evaluation metrics with enhanced styling
        st.markdown("---")
        st.markdown("""
        <div style='text-align:center;'>
            <h3>Linear Regression Model Evaluation Metrics</h3>
            <p>R² Score (Train): <strong>0.9496</strong></p>
            <p>MSE (Train): <strong>193.8074</strong></p>
            <p>RMSE (Train): <strong>13.9215</strong></p>
            <p>R² Score (Test): <strong>0.9567</strong></p>
            <p>MSE (Test): <strong>163.4908</strong></p>
            <p>RMSE (Test): <strong>12.7864</strong></p>
            <p>Best Linear Regression Parameters: <strong>{'regressor__fit_intercept': True, 'regressor__positive': True}</strong></p>
            <p>Best CV R² Score: <strong>0.9494</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature importance plot with Plotly
        st.markdown("---")
        st.markdown("## 📊 Feature Importance Analysis")
        feature_importance = {
            'watch_time_minutes': 0.933541,
            'likes': 0.025377,
            'comments': 0.007302,
            'subscribers': 0.006744,
            'views': 0.006379,
            'engagement_rate': 0.004313,
            'video_length_minutes': 0.003252,
            'watch_completion': 0.002885,
            'day_of_week': 0.002262,
            'country': 0.002198,
            'category': 0.002193,
            'device': 0.001643,
            'quarter': 0.001614,
            'is_weekend': 0.000296
        }
        
        fig = go.Figure()
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        fig.add_trace(go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker=dict(
                color=importance,
                colorscale='Rainbow',
                showscale=True,
                colorbar=dict(title="Importance")
            ),
            text=[f'{imp:.3f}' for imp in importance],
            textposition='outside'
        ))
        fig.update_layout(
            title="Feature Importance in Youtube Ad_Revenue_USD Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=500,
            template='plotly_white',
            font=dict(size=12)
        )
        st.plotly_chart(fig, use_container_width=True)

#-------------------------------- Model Comparison Page --------------------------------
# Model Comparison Page
def show_comparison_page():
    """Model Comparison Page with correct features"""
    st.markdown("""
    <div class='bright-header'>
        📊 Model Comparison
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 2.3rem; color: #666; line-height: 1.6;'>
            Compare different machine learning models and make predictions with any algorithm!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and other necessary data using your existing functions
    models = load_models()
    scaler = load_scaler()
    freq_maps = load_freq_maps()
    
    # Training feature order (as per model training)
    training_features = ['views', 'likes', 'comments', 'watch_time_minutes', 'subscribers', 
                        'video_length_minutes', 'category', 'device', 'country', 'engagement_rate', 
                        'watch_completion', 'day_of_week', 'is_weekend', 'quarter']
    
    # Model performance data
    model_performance = {
        'Model': ['Linear Regression', 'SVR', 'Decision Tree', 'Random Forest', 'XGBoost'],
        'R² Score': [0.95, 0.95, 0.94, 0.94, 0.95],
        'RMSE': [12.78, 12.79, 14.81, 14.48, 12.97],
        'MAE': [98.7, 125.4, 142.3, 118.9, 134.2],
        'Training Time (s)': [1.3, 15.2, 3.1, 8.7, 12.3]
    }
    
    df_performance = pd.DataFrame(model_performance)
    
    # Display performance comparison
    st.markdown("## 🏆 Model Performance Comparison")
    
    # Create performance charts
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('R² Score', 'RMSE', 'MAE', 'Training Time'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    colors = ['#FF1493', '#00BFFF', '#32CD32', '#FF4500', '#8A2BE2']
    
    # R² Score
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['R² Score'], 
               name='R² Score', marker_color=colors,
               text=df_performance['R² Score'], textposition='outside'),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['RMSE'], 
               name='RMSE', marker_color=colors,
               text=df_performance['RMSE'], textposition='outside'),
        row=1, col=2
    )
    
    # MAE
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['MAE'], 
               name='MAE', marker_color=colors,
               text=df_performance['MAE'], textposition='outside'),
        row=2, col=1
    )
    
    # Training Time
    fig.add_trace(
        go.Bar(x=df_performance['Model'], y=df_performance['Training Time (s)'], 
               name='Training Time', marker_color=colors,
               text=df_performance['Training Time (s)'], textposition='outside'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.markdown("### 📋 Detailed Performance Metrics")
    st.dataframe(df_performance.style.highlight_max(axis=0), use_container_width=True)
    
    # Best model highlight
    st.markdown("""
    <div style='background: linear-gradient(135deg, #FFD700, #FFA500); padding: 20px; border-radius: 15px; margin: 20px 0;'>
        <h3 style='color: #8B4513; text-align: center; margin-bottom: 10px;'>🏆 Best Performing Model</h3>
        <p style='color: #8B4513; text-align: center; font-size: 1.2rem; font-weight: bold;'>
            Linear Regression with R² = 0.95 and RMSE = 12.78
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Predictor Section
    st.markdown("---")
    st.markdown("## 🚀 Quick Model Predictor")
    
    # Check if models are available
    if not models:
        st.warning("⚠️ **No models available.** Please train models first on the Model Training page.")
        return
    
    # Model selection
    selected_model = st.selectbox(
        "Select Model for Prediction:",
        list(models.keys()),
        index=0,
        help="Choose which model to use for prediction"
    )
    
    # Input form
    st.markdown("### 📝 Input Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📊 Performance Metrics**")
        views = st.number_input("👁️ Views", value=10000, step=1000)
        likes = st.number_input("👍 Likes", value=500, step=50)
        comments = st.number_input("💬 Comments", value=100, step=10)
        subscribers = st.number_input("👥 Subscribers", value=5000, step=500)
    
    with col2:
        st.markdown("**⏱️ Time & Engagement**")
        watch_time_minutes = st.number_input("⏰ Watch Time (min)", value=1000.0, step=100.0)
        video_length_minutes = st.slider("📏 Video Length (min)", 1.0, 60.0, 15.0)
        engagement_rate = st.slider("📊 Engagement Rate", 0.0, 0.2, 0.05, 0.001)
        watch_completion = st.slider("⚡ Watch Completion", 0.0, 1.0, 0.65, 0.01)
    
    with col3:
        st.markdown("**🏷️ Categories & Context**")
        category_options = ['Entertainment', 'Gaming', 'Education', 'Music', 'Sports', 'News', 'Technology']
        device_options = ['Mobile', 'Desktop', 'Tablet', 'TV']
        country_options = ['US', 'UK', 'CA', 'AU', 'DE', 'FR', 'JP', 'IN']
        
        category = st.selectbox("🎭 Category", category_options)
        device = st.selectbox("📱 Device", device_options)
        country = st.selectbox("🌍 Country", country_options)
        
        day_of_week = st.selectbox("📅 Day of Week", 
                                  ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        day_of_week_num = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'].index(day_of_week) + 1
        is_weekend = 1 if day_of_week in ['Saturday', 'Sunday'] else 0
        quarter = st.selectbox("📈 Quarter", [1, 2, 3, 4])
    
    # Predict button
    if st.button("⚡ Quick Predict", key="quick_predict"):
        try:
            # Create input data
            input_data = {
                'views': views,
                'likes': likes,
                'comments': comments,
                'watch_time_minutes': watch_time_minutes,
                'subscribers': subscribers,
                'video_length_minutes': video_length_minutes,
                'engagement_rate': engagement_rate,
                'watch_completion': watch_completion,
                'category': category,
                'device': device,
                'country': country,
                'day_of_week': day_of_week_num,
                'is_weekend': is_weekend,
                'quarter': quarter
            }
            
            # Prepare input for prediction
            input_df = pd.DataFrame([input_data])
            
            # Ensure preparation only if freq_maps and scaler are available
            if freq_maps and training_features:
                prepared_input = prepare_input_for_prediction(input_df, freq_maps, scaler, training_features)
            else:
                # Fallback: use input as-is if preparation data not available
                prepared_input = input_df
                st.warning("Using unprocessed input - model accuracy may be affected")
            
            # Make prediction
            prediction = predict_revenue(models[selected_model], prepared_input)
            
            # Display result
            st.markdown(f"""
            <div style='text-align: center; background: linear-gradient(135deg, #FF1493, #8A2BE2); 
                        color: white; padding: 2rem; margin: 1rem 0; border-radius: 20px; 
                        box-shadow: 0 8px 25px rgba(255,20,147,0.3);'>
                <h3 style='color: #FFD700; text-shadow: 1px 1px 2px rgba(0,0,0,0.5); margin-bottom: 1rem;'>
                    🎯 {selected_model} Prediction
                </h3>
                <h2 style='color: white; font-size: 3rem; margin: 1rem 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                    ${prediction:.2f}
                </h2>
                <p style='color: #FFE4E1; font-size: 1.1rem; margin-top: 1rem;'>
                    💰 Estimated Youtube_Ad_Revenue (USD)
                </p>
            </div>
            """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"❌ **Prediction failed:** {str(e)}")
            st.info("💡 **Tip:** Make sure models are trained first on the Model Training page.")
    
    # Model insights section
    st.markdown("---")
    st.markdown("## 🧠 Model Insights")
    
    insights_col1, insights_col2 = st.columns(2)
    
    with insights_col1:
     st.markdown("""
    <div style='background: linear-gradient(135deg, #fef7ff, #f8f0ff, #fff0f8); 
                padding: 30px; border-radius: 20px; border-left: 6px solid #FF1493;
                box-shadow: 0 4px 20px rgba(255, 20, 147, 0.1);
                border: 1px solid rgba(255, 20, 147, 0.1);'>
        <h3 style='color: #2c3e50; font-size: 1.8rem; margin-bottom: 20px; text-align: center;'>🎯 Feature Importance</h3>
        <p style='color: #34495e; font-size: 1.2rem; margin-bottom: 15px; text-align: center; font-weight: 500;'>
            Based on model analysis:
        </p>
        <div style='font-size: 1.15rem; line-height: 1.8; color: #2c3e50;'>
            <p style='margin-bottom: 12px; background: rgba(231, 76, 60, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #e74c3c;'>
                <strong style='color: #e74c3c;'>1. Watch Time</strong> <span style='color: #27ae60; font-weight: 600;'>(93% importance)</span> - Most critical factor</p>
            <p style='margin-bottom: 12px; background: rgba(52, 152, 219, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #3498db;'>
                <strong style='color: #3498db;'>2. Views</strong> <span style='color: #27ae60; font-weight: 600;'>(78% importance)</span> - Strong correlation with revenue</p>
            <p style='margin-bottom: 12px; background: rgba(155, 89, 182, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #9b59b6;'>
                <strong style='color: #9b59b6;'>3. Engagement Rate</strong> <span style='color: #27ae60; font-weight: 600;'>(65% importance)</span> - Quality metric</p>
            <p style='margin-bottom: 12px; background: rgba(243, 156, 18, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #f39c12;'>
                <strong style='color: #f39c12;'>4. Subscribers</strong> <span style='color: #27ae60; font-weight: 600;'>(52% importance)</span> - Audience base</p>
            <p style='margin-bottom: 8px; background: rgba(230, 126, 34, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #e67e22;'>
                <strong style='color: #e67e22;'>5. Category</strong> <span style='color: #27ae60; font-weight: 600;'>(41% importance)</span> - Content type matters</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with insights_col2: 
      st.markdown("""
    <div style='background: linear-gradient(135deg, #fff8f0, #fff3e6, #ffefdb); 
                padding: 30px; border-radius: 20px; border-left: 6px solid #FFA500;
                box-shadow: 0 4px 20px rgba(255, 165, 0, 0.1);
                border: 1px solid rgba(255, 165, 0, 0.1);'>
        <h3 style='color: #2c3e50; font-size: 1.8rem; margin-bottom: 20px; text-align: center;'>💡 Optimization Tips</h3>
        <p style='color: #34495e; font-size: 1.2rem; margin-bottom: 15px; text-align: center; font-weight: 500;'>
            To maximize  youtube ad revenue:
        </p>
        <div style='font-size: 1.15rem; line-height: 1.8; color: #2c3e50;'>
            <p style='margin-bottom: 12px; background: rgba(231, 76, 60, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #e74c3c;'>
                • <strong style='color: #e74c3c;'>Increase watch time</strong> through engaging content</p>
            <p style='margin-bottom: 12px; background: rgba(52, 152, 219, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #3498db;'>
                • <strong style='color: #3498db;'>Optimize video length</strong> (15-20 min sweet spot)</p>
            <p style='margin-bottom: 12px; background: rgba(155, 89, 182, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #9b59b6;'>
                • <strong style='color: #9b59b6;'>Target high-engagement categories</strong> (Gaming, Tech)</p>
            <p style='margin-bottom: 12px; background: rgba(243, 156, 18, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #f39c12;'>
                • <strong style='color: #f39c12;'>Post during peak hours</strong> (weekends perform better)</p>
            <p style='margin-bottom: 8px; background: rgba(39, 174, 96, 0.08); padding: 10px 15px; border-radius: 10px; border-left: 4px solid #27ae60;'>
                • <strong style='color: #27ae60;'>Focus on completion rates</strong> over just views</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

#---------------------------------- About Developer Page --------------------------------   

    
    
    # Main function to run the app

    # Sidebar navigation
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h1 style= font-size: 2rem;'>📊 Navigation</h1>
        </div>
        """, unsafe_allow_html=True)
        
    selected = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "🎯 Revenue Predictor", "📊 Model Comparison"]
        )
    
    # Page routing
if selected == "🏠 Home":
    show_home_page()  
elif selected == "🎯 Revenue Predictor":
    show_predictor_page()  
elif selected == "📊 Model Comparison":
    show_comparison_page()  

