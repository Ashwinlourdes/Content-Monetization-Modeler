# Content-Monetization-Modeler
Purpose & business problem

Creators, influencers and media teams need to forecast how much a video will earn (for planning, budgeting and content strategy). This project predicts per-video ad revenue and surfaces the features and optimizations that most affect monetization so stakeholders can make data-driven publishing and content decisions. 


Dataset & features (columns)

The app expects a cleaned dataset of YouTube video analytics. Columns used by the system (explicit in the code) include:

Primary features (used for modeling / input):

views

likes

comments

watch_time_minutes

subscribers

video_length_minutes

category (categorical, frequency-encoded)

device (categorical, frequency-encoded)

country (categorical, frequency-encoded)

engagement_rate — computed as (likes + comments) / views (or provided).

watch_completion (fraction 0–1)

day_of_week (0–6)

is_weekend (0/1)

quarter (1–4)

Target / derived columns that appear in the dataset or are created:

ad_revenue_usd (target)

likes_per_1k_views (derived)

log_ad_revenue (derived: np.log1p(ad_revenue_usd))

(These column names and engineered fields are directly from the uploaded code.) 


Preprocessing & feature engineering (step-by-step)

This section describes the exact transformation pipeline used when preparing inputs for prediction.

Load raw cleaned CSV

The app reads cleaned_data.csv into df. 



Frequency encoding for categorical fields

category, device, and country are converted to numeric representations using pre-computed frequency maps (freq_maps.json). If a map is missing for a category, a default 0 is used. 

Drop target if present

If ad_revenue_usd is provided in input, it is dropped before prediction. 


Ensure feature alignment

The code enforces the same feature order used for training (training_features list). Any missing feature is added with a default value of 0. This prevents misalignment between input and model expectation. 


Fill NaNs and reindex

Any remaining NaN values are filled with 0 and the DataFrame is reindexed to the training order. 


Scaling / normalization

If a scaler.pkl is present, numerical features are scaled using the loaded scaler. If the scaler file is missing, scaling is skipped (app warns). 


Feature engineering (performed in EDA / dataset):

engagement_rate = (likes + comments) / views

likes_per_1k_views = likes / (views / 1000 + 1e-9)

log_ad_revenue = np.log1p(ad_revenue_usd) — suggested for modeling target stabilization (log transform noted in app). 


Notes / practical details

Frequency maps and scaler must be placed where the app expects them or update paths in config. The code contains absolute paths (e.g., D:\Content_Monetization\...) — these should be converted to project-relative paths for portability. 


5 — Models used & why they were chosen

The app loads five pre-trained models from disk:

Linear Regression — chosen for interpretability and fast inference; used as the primary example/model in the UI.

Support Vector Regression (SVR) — robust to outliers in some regimes; used for comparison.

Decision Tree — simple non-linear baseline.

Random Forest — ensemble to capture non-linearities and interactions.

XGBoost — high-performing gradient boosting model for tabular data.

The selection covers: a simple linear model, kernel model, tree model, ensemble trees, and gradient boosting — giving a good mix for comparison of bias/variance and interpretability vs performance tradeoffs. 


6 — Training details (what the code documents vs what’s missing)

What the app documents:

The code and UI mention 5-fold cross-validation (CV) in the modeling approach and discuss evaluation metrics R², RMSE, and MAE. 



Example model performance values shown in the UI (reported for comparison):

Model	R²	RMSE	MAE	Train time (s)
Linear Regression	0.95	12.78	98.7	1.3
SVR	0.95	12.79	125.4	15.2
Decision Tree	0.94	14.81	142.3	3.1
Random Forest	0.94	14.48	118.9	8.7
XGBoost	0.95	12.97	134.2	12.3

(These performance numbers are presented in the Model Comparison page of your app.) 


What the code does not show (gaps / assumptions):

The exact train/test split ratio, random seed, feature selection process, hyperparameter search ranges, and raw training scripts are not present in learn1.py. The app loads trained models from .pkl files but does not include the training notebook or scripts in the code snippet provided. If training logs or notebooks exist, include them in the repo for reproducibility. 


Recommendation: If you want the training details captured in the repo, add the training notebook (train.ipynb) that contains code to reproduce model training, CV, hyperparameter search, and model export steps.

7 — How predictions are made (input → preprocessing → model → output)

User input / uploaded CSV — values for features (views, likes, comments, watch_time_minutes, etc.). The app accepts either user typed inputs in the Streamlit UI or an uploaded CSV (model comparison section). 



Input conversion — inputs are assembled into a pandas DataFrame.

Preprocessing — prepare_input_for_prediction():

Frequency encode categorical fields using freq_maps.json.

Add missing columns and reorder to the training_features order.

Fill NaNs.

Scale numeric features with scaler.pkl (if available). 


Model selection & prediction — the app loads the chosen model (e.g., Linear Regression) and calls model.predict(prepared_input). The predict_revenue() helper contains error handling fallback to a random value if a model is missing or prediction fails. 



Output — the app displays:

Single-video predicted revenue (USD).

Monthly projection (prediction × 4 by default).

Yearly projection (monthly × 12).

Also displays model metrics and a feature importance bar chart. 


8 — Feature importance & interpretation

The app reports feature importance scores (example values shown in the UI). The key findings:

Dominant Predictor: watch_time_minutes — extremely high importance (~0.93), meaning watch time explains most variance in revenue. Improving watch time yields the most revenue impact. 


Secondary Predictors: likes, comments, subscribers, and views — much smaller importance individually but collectively contribute. engagement_rate and watch_completion have lower but non-zero influence. 



Interpretation for stakeholders:

Prioritize strategies that increase watch time and retention (strong hook, mid-video engagement, avoid long low-value segments).

Engagement (likes, comments) is helpful, but does not substitute for sustained watch time from the model’s perspective.

9 — Deployment: how the Streamlit app works (routes/pages)

The Streamlit app provides a sidebar navigation and three main pages (routes) implemented in learn1.py:

🏠 Home (show_home_page)

Project overview, dataset summary, EDA plots, feature engineering notes, modeling approach, and high-level recommendations. 


🎯 Revenue Predictor (show_predictor_page)

Interactive form to enter video metrics, select categorical options (category/device/country from frequency maps), and click Predict to get immediate revenue, monthly, and yearly estimates. Shows model metrics (R², MSE, RMSE). 



📊 Model Comparison (show_comparison_page)

Displays comparative performance metrics for all loaded models with Plotly charts. Supports a “Quick Predict” using any selected model. 

