"""
============================================================
 Content Monetization Modeler – Streamlit App
 GUVI | HCL – Social Media Analytics
 Dataset: youtube_ad_revenue_dataset.csv
============================================================
 Run:  streamlit run streamlit_app.py
============================================================
"""

import os, numpy as np, pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="YouTube Revenue Predictor", page_icon="📺", layout="wide")

CATEGORIES = ["Education","Entertainment","Gaming","Lifestyle","Music","Tech"]
DEVICES    = ["Desktop","Mobile","TV","Tablet"]
COUNTRIES  = ["AU","CA","DE","IN","UK","US"]

MODEL_PATH      = os.path.join("outputs","models","best_model.pkl")
SCALER_PATH     = os.path.join("outputs","models","scaler.pkl")
ENCODER_PATH    = os.path.join("outputs","models","encoders.pkl")
FEATURE_PATH    = os.path.join("outputs","models","feature_cols.pkl")
DATA_PATH       = os.path.join("outputs","cleaned_dataset.csv")
COMPARISON_PATH = os.path.join("outputs","model_comparison.csv")
PLOTS_DIR       = os.path.join("outputs","plots")

@st.cache_resource(show_spinner=False)
def load_artefacts():
    return (joblib.load(MODEL_PATH), joblib.load(SCALER_PATH),
            joblib.load(ENCODER_PATH), joblib.load(FEATURE_PATH))

@st.cache_data(show_spinner=False)
def load_data():       return pd.read_csv(DATA_PATH)

@st.cache_data(show_spinner=False)
def load_comparison(): return pd.read_csv(COMPARISON_PATH)

def build_feature_row(views,likes,comments,watch_time,video_length,
                      subscribers,category,device,country,
                      month,day_of_week,quarter,encoders,feature_cols):
    row = {
        "log_views":            np.log1p(views),
        "likes":                likes,
        "comments":             comments,
        "log_watch_time":       np.log1p(watch_time),
        "video_length_minutes": video_length,
        "log_subscribers":      np.log1p(subscribers),
        "engagement_rate":      (likes+comments)/(views+1),
        "avg_watch_ratio":      watch_time/(views*video_length+1),
        "likes_per_view":       likes/(views+1),
        "comments_per_view":    comments/(views+1),
        "month":                month,
        "day_of_week":          day_of_week,
        "quarter":              quarter,
        "category_enc":         int(encoders["category"].transform([category])[0]),
        "device_enc":           int(encoders["device"].transform([device])[0]),
        "country_enc":          int(encoders["country"].transform([country])[0]),
    }
    return pd.DataFrame([row])[feature_cols]

st.sidebar.title("📺 Content Monetization Modeler")
st.sidebar.markdown("*GUVI | HCL – Social Media Analytics*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate",["🏠 Home","🔍 EDA & Insights","📊 Model Comparison","💰 Revenue Predictor"])
artefacts_ready = all(os.path.exists(p) for p in [MODEL_PATH,SCALER_PATH,ENCODER_PATH,FEATURE_PATH])

if page == "🏠 Home":
    st.title("📺 YouTube Ad Revenue Predictor")
    st.markdown("### GUVI | HCL – Content Monetization Modeler")
    st.markdown("---")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Dataset Rows","122,400"); c2.metric("Features","16")
    c3.metric("Models Trained","5");     c4.metric("Best R²","0.9999")
    st.markdown("""
---
### 📌 What This App Does
Predicts **YouTube ad revenue (USD)** using a Gradient Boosting model.

### 📋 Dataset — `youtube_ad_revenue_dataset.csv`
| Column | Description |
|--------|-------------|
| views | Total view count |
| likes | Total likes |
| comments | Total comments |
| watch_time_minutes | Cumulative watch time |
| video_length_minutes | Duration |
| subscribers | Channel subscribers |
| category | Education / Entertainment / Gaming / Lifestyle / Music / Tech |
| device | Desktop / Mobile / TV / Tablet |
| country | AU / CA / DE / IN / UK / US |
| **ad_revenue_usd** | 🎯 Target variable |

### 🤖 Model Results
| Model | R² |
|---|---|
| Linear Regression | ~0.97 |
| Ridge Regression | ~0.97 |
| Decision Tree | ~0.99 |
| Random Forest | ~0.9999 |
| **Gradient Boosting ✅** | **~0.9999** |
""")
    if not artefacts_ready: st.warning("⚠️  Run all cells in the Jupyter notebook first.")
    else: st.success("✅  Model artefacts loaded. Go to Revenue Predictor to predict!")

elif page == "🔍 EDA & Insights":
    st.title("🔍 Exploratory Data Analysis")
    if not os.path.exists(DATA_PATH): st.error("Run the notebook first."); st.stop()
    df = load_data()
    st.markdown(f"**Cleaned dataset:** `{df.shape[0]:,}` rows × `{df.shape[1]}` columns")
    with st.expander("📊 Descriptive Statistics"): st.dataframe(df.describe().round(3))
    plot_titles = {
        "01_revenue_distribution.png":"1. Revenue Distribution",
        "02_revenue_by_category.png":"2. Revenue by Category",
        "03_revenue_by_country.png":"3. Revenue by Country",
        "04_revenue_by_device.png":"4. Revenue by Device",
        "05_correlation_heatmap.png":"5. Correlation Heatmap",
        "06_scatter_plots.png":"6. Scatter Plots",
        "07_monthly_trend.png":"7. Monthly Trend",
        "08_outlier_boxplots.png":"8. Outlier Boxplots",
        "09_engineered_features.png":"9. Engineered Features",
        "10_model_comparison.png":"10. Model Comparison",
        "11_actual_vs_predicted.png":"11. Actual vs Predicted",
        "12_residuals.png":"12. Residuals",
        "13_feature_importance.png":"13. Feature Importance",
    }
    if os.path.exists(PLOTS_DIR):
        avail = [f for f in sorted(os.listdir(PLOTS_DIR)) if f.endswith(".png") and f in plot_titles]
        for i in range(0, len(avail), 2):
            cols = st.columns(2)
            for j,col in enumerate(cols):
                if i+j < len(avail):
                    col.subheader(plot_titles[avail[i+j]])
                    col.image(os.path.join(PLOTS_DIR, avail[i+j]))
    st.subheader("💡 Key Stats")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Min",    f"${df['ad_revenue_usd'].min():.2f}")
    c2.metric("Median", f"${df['ad_revenue_usd'].median():.2f}")
    c3.metric("Mean",   f"${df['ad_revenue_usd'].mean():.2f}")
    c4.metric("Max",    f"${df['ad_revenue_usd'].max():.2f}")

elif page == "📊 Model Comparison":
    st.title("📊 Model Performance Comparison")
    if not os.path.exists(COMPARISON_PATH): st.error("Run the notebook first."); st.stop()
    cdf = load_comparison()
    mc  = "R²" if "R²" in cdf.columns else "R2"
    st.dataframe(cdf.style.highlight_max(subset=[mc],color="#c6efce")
                          .highlight_min(subset=["RMSE","MAE"],color="#c6efce"), use_container_width=True)
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    for ax,metric,color in zip(axes,[mc,"RMSE","MAE"],["#4C72B0","#C44E52","#55A868"]):
        ax.barh(cdf["Model"],cdf[metric],color=color,edgecolor="white")
        ax.set(title=metric,xlabel=metric); ax.invert_yaxis()
    plt.suptitle("Model Comparison",fontsize=13,fontweight="bold"); plt.tight_layout(); st.pyplot(fig)
    best = cdf.sort_values(mc,ascending=False).iloc[0]
    st.success(f"🏆 Best: **{best['Model']}** | R²={best[mc]} | RMSE={best['RMSE']} | MAE={best['MAE']}")

elif page == "💰 Revenue Predictor":
    st.title("💰 Predict Ad Revenue")
    if not artefacts_ready: st.error("Run the notebook first."); st.stop()
    model,scaler,encoders,feature_cols = load_artefacts()
    st.markdown("---")
    col1,col2,col3 = st.columns(3)
    with col1:
        st.subheader("📈 Engagement")
        views       = st.number_input("Views",       9000,  11000,  10000, 100)
        likes       = st.number_input("Likes",        100,   2500,   1100,  50)
        comments    = st.number_input("Comments",      10,   1000,    200,  10)
        subscribers = st.number_input("Subscribers", 1000, 999999, 500000, 10000)
    with col2:
        st.subheader("⏱️ Watch Metrics")
        watch_time   = st.number_input("Watch Time (min)", 1000.0, 200000.0, 30000.0, 1000.0)
        video_length = st.number_input("Video Length (min)", 1.0, 60.0, 10.0, 0.5)
        month        = st.slider("Month", 1, 12, 6)
        day_of_week  = st.slider("Day of Week (0=Mon)", 0, 6, 2)
        quarter      = st.selectbox("Quarter", [1,2,3,4], index=1)
    with col3:
        st.subheader("🌍 Content Info")
        category = st.selectbox("Category", CATEGORIES)
        device   = st.selectbox("Device",   DEVICES)
        country  = st.selectbox("Country",  COUNTRIES)
    st.markdown("---")
    if st.button("🚀 Predict Revenue", use_container_width=True):
        try:
            row  = build_feature_row(views,likes,comments,watch_time,video_length,
                                     subscribers,category,device,country,
                                     month,day_of_week,quarter,encoders,feature_cols)
            pred = max(0.0, float(model.predict(scaler.transform(row))[0]))
            st.markdown(f"""<div style="background:#e8f5e9;padding:28px;border-radius:14px;text-align:center;">
                <h2 style="color:#2e7d32;margin:0;">💵 Estimated Ad Revenue</h2>
                <h1 style="color:#1b5e20;font-size:3.2rem;margin:10px 0;">${pred:,.2f} USD</h1>
            </div>""", unsafe_allow_html=True)
            eng = (likes+comments)/(views+1)
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Views",f"{views:,}"); c2.metric("Engagement",f"{eng:.3%}")
            c3.metric("Watch/View",f"{watch_time/(views+1):.1f} min"); c4.metric("Category",category)
            st.info(f"📌 **{category}** from **{country}** on **{device}** | {views:,} views | {eng:.2%} engagement → **${pred:,.2f} USD**")
        except Exception as e: st.error(f"Error: {e}")
    st.markdown("---")
    st.caption("Content Monetization Modeler · GUVI | HCL · Dataset: youtube_ad_revenue_dataset.csv")
