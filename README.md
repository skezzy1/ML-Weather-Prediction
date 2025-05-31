# 🌦️ Weather Prediction using Machine Learning

This project focuses on forecasting **precipitation levels** across Ukrainian cities using several machine learning algorithms, including Random Forest, Decision Tree, Polynomial Regression, SVR, LightGBM, and XGBoost.

---

## 📌 Objective

- To improve the accuracy of precipitation forecasting.
- To analyze how temperature, wind speed, snow depth, month, and region affect rainfall levels.
- To divide the dataset by **region and season** to improve prediction quality.
- To evaluate feature correlation and multicollinearity.
- To compare multiple regression algorithms.

---

## 🧠 Applied Models

| Model                  | Type        | Key Use |
|------------------------|-------------|----------|
| **Random Forest**      | Ensemble    | Robust prediction using many decision trees |
| **Decision Tree**      | Tree-based  | Easy to interpret, suitable for small regions |
| **Polynomial Regression** | Regression | Captures non-linear patterns |
| **Support Vector Regression (SVR)** | Kernel-based | Suitable for complex nonlinear relations |
| **LightGBM**           | Gradient Boosting | Fast and efficient on large datasets |
| **XGBoost**            | Gradient Boosting | High performance with boosting techniques |

---

## 📁 Project Structure
├── data_preparation/
│ ├── csv/ # Raw and processed datasets
│ ├── regions_seasons/ # Region and season-based subsets
│ ├── outliers/ # Outlier removal scripts
│
├── models/
│ ├── RandomForest/
│ ├── PolynomialAndDesionTree/
│ ├── SVR/
│ └── LightGBM_XGBoost/
│
├── Correlation/
│ └── correlation_result.txt # Correlation matrix
│
├── plots/ # Visualization outputs
└── README.md


---

## 🧼 Data Preprocessing

- **Outlier removal** using Z-score (threshold ±3).
- **Scaling**:
  - StandardScaler for temperature features.
  - MinMaxScaler for wind speed, precipitation, snow depth.
- **Encoding**:
  - One-Hot Encoding for `City`.
  - Region & Season columns added for better granularity.
- **Multicollinearity**:
  - Dropped features with high pairwise correlation (e.g., Min Temperature).

---

## 📊 Evaluation Metrics

- `MSE` (Mean Squared Error)
- `MAE` (Mean Absolute Error)
- `R² Score` (Coefficient of Determination)

---

## 📈 Sample Results

| Model              | R² Score | MSE   | MAE   |
|-------------------|----------|-------|-------|
| Random Forest      | 0.77     | 0.00  | 0.04  |
| Decision Tree      | 0.74     | 0.00  | 0.04  |
| Polynomial Reg.    | 0.64     | 0.00  | 0.05  |
| SVR (RBF kernel)   | 0.83     | 0.00  | 0.03  |
| LightGBM           | 0.79     | 0.00  | 0.04  |
| XGBoost            | 0.75     | 0.00  | 0.04  |
---

## 🔍 Relevance

Recent technologies such as **NVIDIA CorrDiff** highlight the growing importance of precise and adaptive weather forecasting. CorrDiff is a diffusion model that combines **deep learning** with **generative modeling** for highly accurate precipitation simulations. This research aligns with the global demand for accurate, explainable climate models.

---

## ▶️ How to Run

1. Install required packages:
```bash
pip install -r requirements.txt
```
2. Preprocess the data:
```bash
python data_preparation/outliers/preprocess_outliers.py
```
3. Train and evaluate models:
```bash
python models/RandomForest/RandomForestModel.py
```

---

## 👤 Author
- Serhii Herasymchuk
- Faculty of Computer Science, Specialty 124 – Data Science
- Graduation Year: 2026