# E-Commerce_Conversion_Prediction_using_Decision_Trees

## 📌 Project Overview

This project focuses on predicting whether an online visitor will complete a purchase based on their browsing behavior during a session.

The solution uses a **Decision Tree Classifier** with **Pre-Pruning** and **Cost Complexity Post-Pruning** techniques to improve generalization and reduce overfitting.

The dataset contains information about visitor interactions such as:

- Product page visits
- Time spent on pages
- Bounce rates
- Exit rates
- Visitor type
- Traffic source
- Seasonal effects

The target variable is:

🎯 **Revenue**
- True → Purchase Made
- False → No Purchase

---

## 🎯 Business Problem

E-commerce companies receive thousands of visitors daily but only a small percentage complete a purchase.

The objective is to:

✅ Identify high-intent customers

✅ Improve marketing effectiveness

✅ Increase conversion rates

✅ Reduce customer acquisition costs

---

## 🏗️ Project Workflow

### 1️⃣ Data Collection

- Load customer session dataset
- Inspect structure and feature types

### 2️⃣ Data Cleaning

- Missing value treatment
- Duplicate removal

### 3️⃣ Exploratory Data Analysis (EDA)

- Class distribution analysis
- Correlation analysis
- Feature behavior investigation

### 4️⃣ Feature Engineering

- One-Hot Encoding
- Target preparation

### 5️⃣ Model Development

- Baseline Decision Tree
- Pre-Pruning
- Cost Complexity Pruning

### 6️⃣ Hyperparameter Optimization

- GridSearchCV
- Cross Validation

### 7️⃣ Model Evaluation

- Classification Report
- Precision
- Recall
- F1 Score

### 8️⃣ Feature Importance Analysis

- Identify key drivers of customer purchases

---

## 📊 Dataset Features

| Feature | Description |
|----------|-------------|
| Administrative | Account-related page visits |
| Administrative_Duration | Time spent on administrative pages |
| Informational | Informational page visits |
| Informational_Duration | Time spent on informational pages |
| ProductRelated | Product page visits |
| ProductRelated_Duration | Time spent on product pages |
| BounceRates | Single-page exit percentage |
| ExitRates | Exit percentage |
| PageValues | Value of pages before transaction |
| SpecialDay | Proximity to special occasions |
| Month | Month of visit |
| OperatingSystems | User operating system |
| Browser | Browser type |
| Region | Geographic region |
| TrafficType | Traffic source |
| VisitorType | Returning/New Visitor |
| Weekend | Weekend visit indicator |
| Revenue | Target Variable |

---

## 🤖 Machine Learning Model

### Decision Tree Classifier

The model was selected because:

- Handles non-linear relationships
- Requires minimal preprocessing
- Provides interpretability
- Handles mixed feature types effectively

---

## ✂️ Pruning Strategy

### Pre-Pruning

```python
max_depth=5
min_samples_split=20
min_samples_leaf=10
```

Purpose:

- Reduce overfitting
- Improve generalization
- Prevent excessively complex trees

### Post-Pruning

Cost Complexity Pruning:

```python
ccp_alpha
```

Purpose:

- Remove weak branches
- Simplify decision rules
- Improve model robustness

---

## 📈 Evaluation Metric

Since the dataset is imbalanced, the primary evaluation metric is:

### F1 Score

F1 Score balances:

- Precision
- Recall

Formula:

F1 = 2 × (Precision × Recall) / (Precision + Recall)

Why not Accuracy?

Because a model can achieve high accuracy by simply predicting the majority class while failing to identify actual buyers.

---

## 🔍 Feature Importance

The model identifies which features contribute most to purchase prediction.

Examples:

- PageValues
- ProductRelated
- ProductRelated_Duration
- BounceRates
- ExitRates
- VisitorType

---

## 🛠️ Technologies Used

### Programming Language

- Python 🐍

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📂 Project Structure

```
ShopSmart-Purchase-Prediction/
│
├── data/
│   └── shop_smart_ecommerce.csv
│
├── notebooks/
│   └── ShopSmart_Project.ipynb
│
├── src/
│   └── model_training.py
│
├── README.md
│
└── requirements.txt
```

---

## 🚀 Future Improvements

- Random Forest
- XGBoost
- LightGBM
- Feature Engineering
- SMOTE for class balancing
- Model Explainability using SHAP

---

## 📊 Business Insights

The project helps businesses:

- Identify likely buyers
- Improve customer targeting
- Increase revenue
- Optimize marketing spend
- Improve conversion rates

---
