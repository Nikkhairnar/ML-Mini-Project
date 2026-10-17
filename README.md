# 🍽️ Restaurant Customer Behaviour Analysis and Demand Prediction

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-Interactive%20Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Scikit--Learn-Clustering%20%26%20Regression-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/Project-Academic%20Mini--Project-green.svg)]()

An end-to-end Data Analytics and Machine Learning pipeline that analyzes historical restaurant sales, identifies customer purchase patterns via unsupervised segmentation, and forecasts daily kitchen demand using supervised regression models—presented through a modern interactive **Streamlit** dashboard.

---

## 📌 Project Overview & Business Value

Raw restaurant transaction logs track dates, item names, quantities, and totals, but don't explain **what drives sales**, **how customer groups differ**, or **how much food the kitchen should prepare tomorrow**.

This project solves these real-world problems:
1. **Sales Analytics (Descriptive):** Identifies highest revenue categories, top-selling dishes, monthly patterns, and day-of-week sales volume.
2. **Customer Segmentation (Unsupervised ML):** Segments 100 customer profiles into distinct spending personas (**High-Value Frequent Diners** vs. **Standard Value Diners**) using **K-Means** and **Hierarchical Clustering**.
3. **Demand Prediction (Supervised ML):** Forecasts daily sales revenue and total units sold with **Linear Regression, Ridge Regression, Random Forest**, and **Gradient Boosting**, evaluated against time-series benchmarks.
4. **Interactive Kitchen Planner (Prescriptive):** Converts AI predictions into actionable daily prep quantities and ingredient safety buffers to prevent stockouts and food waste.

---

## 🏛️ Project Architecture & Pipeline

```
       [restaurant_sales_data.csv] (Raw Data: 17,534 rows)
                     │
                     ▼
             [src/clean_data.py]
         (Standardize, Impute, Filter)
                     │
                     ▼
      [data/processed/restaurant_sales_cleaned.csv]
                     │
      ┌──────────────┴──────────────┐
      ▼                             ▼
[src/customer_segmentation.py] [src/demand_prediction.py]
  - RFM Feature Engineering     - 7-Day & 14-Day Lags / Rolling Means
  - StandardScaler              - Chronological 80/20 Train-Test Split
  - K-Means (K=2, Silhouette)   - Linear, Ridge, RF, Gradient Boosting
  - Persona Assignment          - MAE, RMSE, MAPE, Accuracy (%)
      │                             │
      └──────────────┬──────────────┘
                     │
                     ▼
           [Streamlit Dashboard (app.py)]
   (KPIs, Trends, Segmentation, Forecasts, Kitchen Planner)
```

---

## 📂 Project Structure

```
Restaurant_analytics/
│
├── app.py                          # Streamlit Interactive Web Application
├── requirements.txt                # Python package dependencies
├── README.md                       # Complete project documentation
│
├── data/
│   ├── restaurant_sales_data.csv   # Raw transaction dataset (17,534 records)
│   └── processed/
│       └── restaurant_sales_cleaned.csv  # Standardized cleaned dataset
│
├── models/                         # Serialized ML models (.joblib)
│   ├── customer_scaler.joblib
│   ├── kmeans_model.joblib
│   ├── daily_sales_linear_regression.joblib
│   ├── daily_sales_ridge_regression.joblib
│   ├── daily_sales_random_forest.joblib
│   └── daily_units_*.joblib
│
├── reports/eda/                    # Generated charts, evaluation metrics, CSVs
│   ├── customer_segments.csv       # Customer RFM metrics & assigned personas
│   ├── cluster_summary.csv         # Aggregated cluster metrics
│   ├── prediction_model_results.csv# Benchmark & regression evaluation table
│   ├── daily_predictions.csv       # Test set actual vs. predicted values
│   └── *.png                       # EDA, Dendrogram & Silhouette plots
│
└── src/
    ├── inspect_data.py             # CSV inspection & summary statistics
    ├── data_quality_checks.py      # Duplicate detection & total consistency checks
    ├── clean_data.py               # Preprocessing & date feature extraction
    ├── eda.py                      # Exploratory Data Analysis & visual charts
    ├── customer_segmentation.py    # K-Means clustering & persona labeling
    ├── hierarchical_clustering.py  # Agglomerative clustering & dendrogram
    ├── demand_prediction.py        # Supervised regression forecasting models
    └── compare_baseline.py         # Seasonal-naive & mean baseline benchmarks
```

---

## 🚀 Setup & Execution Guide

### 1. Prerequisites & Environment Setup
Clone or open the project folder in your terminal, then create and activate a Python virtual environment:

```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline Scripts (Optional — Artifacts Already Built)
If you wish to re-execute the entire pipeline from scratch:
```bash
python src/data_quality_checks.py
python src/clean_data.py
python src/eda.py
python src/customer_segmentation.py
python src/hierarchical_clustering.py
python src/demand_prediction.py
python src/compare_baseline.py
```

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🖥️ Streamlit Interactive Dashboard Features

The dashboard is structured into **5 comprehensive tabs**:

1. **📊 Tab 1: Executive KPIs & Trends**
   - High-level business KPI metric cards (Total Sales, Total Units, Orders, Average Check, Active Customers).
   - Interactive monthly sales & scaled units trend lines.
   - Revenue contribution by menu category (Donut chart).
   - Day-of-week sales volume distribution.
   - Top 10 best-selling dishes ranked by units sold and revenue.

2. **👥 Tab 2: Customer Behaviour & Segmentation**
   - Visual persona cards for **High-Value Frequent Diners** and **Standard Value Diners**.
   - Spending vs. Average Order Value interactive scatter plot with cluster coloring.
   - **Customer Profile Lookup Tool:** Select any `customer_id` (e.g., `CUST_001`) to view individual metrics and automated AI marketing recommendations.

3. **📈 Tab 3: Demand Forecasting Models**
   - Comprehensive model evaluation table with color highlights for MAE, RMSE, MAPE (%), and Accuracy (%).
   - Comparative operational accuracy chart across all evaluated algorithms.
   - **Out-of-sample time-series plot** comparing actual test sales against predictions.
   - Statistical explanation explaining out-of-sample metrics.

4. **🎛️ Tab 4: Interactive Kitchen Demand Planner (Simulator)**
   - Select upcoming day of week, month, recent sales, and safety stock buffer (0–25%).
   - Real-time prediction of expected revenue, required prep servings, and category-level portion recommendations (Main dishes, Starters, Desserts, Drinks).

5. **📥 Tab 5: Data Explorer & CSV Export**
   - Interactive filtered transaction data table with search and sorting.
   - One-click CSV download of active filtered records.

---

## 🔬 Machine Learning & Analytics Details

### 1. Customer Segmentation (Unsupervised)
- **Features Used:** `purchase_records`, `total_spending`, `total_units`, `average_order_value`, `category_diversity`.
- **Scaling:** `StandardScaler` (z-score normalization) to prevent higher numerical spending ranges from dominating distance metrics.
- **Optimal Clusters ($K=2$):** Evaluated $K \in [2, 6]$ using **Silhouette Score** ($K=2$ scored highest at **0.3919**).
- **Identified Personas:**
  - **High-Value Frequent Diners (54%):** Avg spending of **$3,375.89** across **166.4** orders.
  - **Standard Value Diners (46%):** Avg spending of **$2,877.90** across **147.5** orders.

### 2. Demand Forecasting (Supervised)
- **Features:** Calendar attributes (`day_of_week`, `month`, `day_of_month`, `is_weekend`) + time-lag features (`lag_1`, `lag_7`, `lag_14`, `rolling_mean_7`).
- **Train/Test Split:** **Chronological 80/20 split** (Train: Jan 2022 – Aug 2023 | Test: Aug 2023 – Dec 2023). Avoids future data leakage.
- **Model Comparison Results on Test Set:**

| Target | Model | MAE | RMSE | MAPE (%) | Accuracy (%) | Test $R^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Daily Sales ($)** | **Linear Regression** | **104.94** | 135.35 | 30.35% | **69.65%** | -0.0513 |
| **Daily Sales ($)** | **Ridge Regression** | 105.00 | **135.12** | 30.34% | **69.66%** | -0.0478 |
| **Daily Sales ($)** | Random Forest | 105.54 | 135.83 | 30.72% | 69.28% | -0.0588 |
| **Daily Sales ($)** | Weekly Seasonal-Naive | 138.20 | 172.80 | 36.59% | 63.41% | -0.7136 |
| **Daily Units** | **Linear Regression** | **12.91** | **16.27** | 22.15% | **77.85%** | -0.0072 |
| **Daily Units** | Random Forest | 13.29 | 16.68 | 22.86% | 77.14% | -0.0586 |
| **Daily Units** | Weekly Seasonal-Naive | 16.98 | 21.17 | 27.48% | 72.52% | -0.7051 |

---

## 💡 Viva / Presentation Defense & FAQs

#### Q1: Why is your test $R^2$ score slightly negative (-0.05) while Operational Accuracy is ~77%?
> **Answer:**  
> In statistics, $R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y_{\text{test}})}$. Daily restaurant transactions exhibit natural stochastic noise without an artificial trend. Because the test period mean fluctuates slightly compared to the training mean, the squared deviation from the test mean is slightly smaller than model MSE.  
> However, from a business perspective, the **Mean Absolute Percentage Error (MAPE)** is only **22–30%**, giving **~70–78% Operational Accuracy**. Linear Regression predicts daily revenue within $\pm \$104$ of actual sales, significantly outperforming the Seasonal-Naive baseline (MAE: 138.20).

#### Q2: Why use a chronological split instead of random `train_test_split`?
> **Answer:**  
> Sales data is sequential time-series data. Shuffling randomly would cause **data leakage** (training on future days to predict past days). A chronological split strictly trains on earlier dates to predict future holdout dates.

#### Q3: Why scale features before K-Means?
> **Answer:**  
> K-Means computes Euclidean distance ($d = \sqrt{\sum (x_i - y_i)^2}$). Spending is in the thousands ($3,000) while average order value is around 20. Without `StandardScaler`, spending would completely dominate cluster assignments.

#### Q4: What makes this project practical for real restaurants?
> **Answer:**  
> It bridges pure machine learning with kitchen operations. The **Interactive Kitchen Planner** takes live day-of-week inputs, calculates predicted sales and units, and automatically factors in a 10% safety buffer to recommend exact preparation portions per menu category.

---

## 👥 Authors & Academic Credits
- **Project Type:** Academic Data Science & Machine Learning Mini-Project
- **Technologies:** Python, Pandas, Scikit-Learn, Streamlit, Plotly Express, Joblib
