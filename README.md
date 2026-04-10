# Customer Churn Prediction

A machine learning web app that predicts whether a telecom customer will churn — built end-to-end from raw data to deployed application.

**[Live Demo → Click Here](https://churnprediction-jwedc2rq3lmg78nc8hw6gw.streamlit.app/)**

---

##  Problem Statement

Telecom companies lose millions every year to customer churn. Acquiring a new customer costs **5–7x more** than retaining an existing one. This project builds a predictive system that identifies at-risk customers before they leave — giving businesses time to intervene.

---

## Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** Binary classification — Churn (Yes/No)
- **Class imbalance:** 73% No Churn / 27% Churn

---

##  Approach

Raw Data → EDA → Preprocessing → SMOTE → Model Training → Evaluation → Deployment

### Key Steps:
- **EDA** — Identified top churn drivers: Contract type, Tenure, Monthly Charges
- **Preprocessing** — Label encoding, One-Hot encoding, StandardScaler
- **Class Imbalance** — Handled using SMOTE (Synthetic Minority Oversampling)
- **Model Comparison** — Logistic Regression vs Random Forest vs XGBoost
- **Tuning** — RandomizedSearchCV with 5-fold cross validation
- **Deployment** — Streamlit web app hosted on Streamlit Cloud

---

##  Model Performance

| Model | Accuracy | F1 Score | Recall | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression | 0.75 | 0.56 | 0.62 | 0.83 |
| Random Forest | 0.80 | 0.61 | 0.65 | 0.85 |
| **XGBoost ** | **0.82** | **0.65** | **0.71** | **0.87** |

> **Why AUC-ROC over Accuracy?** Data is imbalanced — accuracy is misleading. AUC-ROC measures the model's ability to distinguish churners from non-churners regardless of threshold.

> **Why Recall matters?** Missing a churner costs far more than wrongly flagging a loyal customer. High recall = catching more actual churners.

---

##  Key Insights from EDA

-  **Month-to-month** customers churn at **3x the rate** of yearly contract customers
-  **First 12 months** are the highest risk period — new customers churn most
-  Churners pay **~$15 more per month** on average than retained customers
-  **Fiber optic** internet users churn significantly more than DSL users

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Visualizations |
| Scikit-learn | Preprocessing + Modeling |
| XGBoost | Final model |
| Imbalanced-learn | SMOTE for class imbalance |
| Streamlit | Web app deployment |
| GitHub | Version control |

---

##  Project Structure

churn-prediction/
├── app/
│   └── app.py              ← Streamlit web app
├── data/
│   └── telco_churn.csv     ← Raw dataset
├── models/
│   ├── model.pkl           ← Trained XGBoost model
│   └── scaler.pkl          ← Fitted StandardScaler
├── notebooks/
│   └── EDA.ipynb           ← Full analysis + modeling
├── requirements.txt
└── README.md

---

##  Run Locally

git clone https://github.com/mayanktiwari-cmd/churn_prediction.git
cd churn_prediction
pip install -r requirements.txt
streamlit run app/app.py

---

##  What I Learned

- Handling real-world class imbalance with SMOTE
- Why evaluation metrics matter more than accuracy
- Building end-to-end ML pipelines from data to deployment
- How business context shapes model design decisions

---

##  Author

**Mayank Tiwari**
[GitHub](https://github.com/mayanktiwari-cmd) • [LinkedIn](linkedin.com/in/mayank-tiwari-4916a8373)
