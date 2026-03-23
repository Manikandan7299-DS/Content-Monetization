#  Content Monetization Modeler
**GUVI | HCL – Social Media Analytics Capstone Project**

---

##  Project Overview

A machine-learning pipeline that predicts **YouTube ad revenue (USD)** for individual videos based on performance and contextual features.

The project includes full EDA, preprocessing, feature engineering, 5 regression models, and an interactive Streamlit web app.

---

##  Project Structure

```
content-monetization-modeler/
│
├── Content_Monetization_Modeler.ipynb   ← Full ML pipeline notebook
├── streamlit_app.py                     ← Interactive Streamlit web app
├── requirements.txt                     ← Python dependencies
├── README.md                            ← This file
└── youtube_ad_revenue_dataset.csv       ← Dataset (place in same folder)
```

---

##  Dataset — `youtube_ad_revenue_dataset.csv`

| Column | Type | Description |
|--------|------|-------------|
| `video_id` | str | Unique video identifier |
| `date` | datetime | Video published date |
| `views` | int | Total view count |
| `likes` | float | Total likes |
| `comments` | float | Total comments |
| `watch_time_minutes` | float | Total watch time in minutes |
| `video_length_minutes` | float | Video duration |
| `subscribers` | int | Channel subscriber count |
| `category` | str | Education / Entertainment / Gaming / Lifestyle / Music / Tech |
| `device` | str | Desktop / Mobile / TV / Tablet |
| `country` | str | AU / CA / DE / IN / UK / US |
| `ad_revenue_usd` | float | 🎯 **Target — Ad revenue in USD** |

- **Total Rows:** 122,400
- **Missing Values:** likes, comments, watch_time_minutes (~5%)
- **Duplicates:** ~2,400 rows

---

##  Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/content-monetization-modeler.git
cd content-monetization-modeler

# 2. Install dependencies
pip install -r requirements.txt
```

---

##  How to Run

### Step 1 — Open Jupyter Notebook
```bash
jupyter notebook Content_Monetization_Modeler.ipynb
```
Then: **Kernel → Restart & Run All**

This will:
- Load and clean the dataset
- Perform EDA and save 13 plots
- Engineer 16 features
- Train 5 regression models
- Save best model artefacts to `outputs/`

### Step 2 — Launch Streamlit App
```bash
streamlit run streamlit_app.py
```
Open browser at → **http://localhost:8501**

---

##  Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `engagement_rate` | (likes + comments) / views | Audience interaction |
| `avg_watch_ratio` | watch_time / (views × length) | Retention quality |
| `likes_per_view` | likes / views | Like engagement |
| `comments_per_view` | comments / views | Comment engagement |
| `log_views` | log(1 + views) | Reduce skewness |
| `log_subscribers` | log(1 + subscribers) | Reduce skewness |
| `log_watch_time` | log(1 + watch_time) | Reduce skewness |
| `month` | from date | Seasonal signal |
| `day_of_week` | from date | Weekly pattern |
| `quarter` | from date | Quarterly signal |

---

##  Models & Results

| Model | R² | RMSE | MAE |
|-------|----|------|-----|
| Linear Regression | 0.9712 | 10.54 | 8.99 |
| Ridge Regression | 0.9712 | 10.54 | 8.99 |
| Decision Tree | 0.9995 | 1.44 | 1.15 |
| Random Forest | 0.9999 | 0.60 | 0.44 |
| **Gradient Boosting ✅** | **0.9999** | **0.52** | **0.41** |

> **Best Model: Gradient Boosting Regressor**

---

##  Streamlit App Pages

| Page | Description |
|------|-------------|
| 🏠 Home | Project overview and dataset summary |
| 🔍 EDA & Insights | 13 interactive charts and statistics |
| 📊 Model Comparison | Side-by-side metrics for all 5 models |
| 💰 Revenue Predictor | Enter video metrics → get predicted revenue |

---

##  Evaluation Metrics

- **R² Score** — Proportion of variance explained (higher = better)
- **RMSE** — Root Mean Squared Error (lower = better)
- **MAE** — Mean Absolute Error (lower = better)

---

##  Key Insights

1. **Watch time** is the strongest predictor of ad revenue
2. **Tech** and **Education** categories earn the highest revenue
3. **US** and **CA** viewers generate the most revenue
4. **Engagement rate** adds meaningful signal beyond raw views
5. **Gradient Boosting** achieved R² = 0.9999 on test data

---

##  Dependencies

```
numpy
pandas
matplotlib
seaborn
scikit-learn
joblib
streamlit
```

---

##  Author

Built for the ** Content Monetization Modeler** Capstone Project.

> Dataset: youtube_ad_revenue_dataset.csv — 122,400 YouTube video records
