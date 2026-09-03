# 🧠 Stroke Risk Prediction Analysis - Multinomial Logistic Regression

A data science project focused on cleaning, preprocessing, analyzing, and predicting multi-factor stroke risk levels (`Low`, `Moderate`, `High`) using Multinomial Logistic Regression.

---

## 📌 Overview

Stroke risk assessment involves complex interactions between biological metrics, lifestyle habits, and medical history. This repository contains end-to-end data processing, feature engineering routines, and model training scripts to evaluate patient profiles and accurately classify stroke risk tiers.

---

## 📊 Dataset Overview

The project utilizes the `stroke_risk_prediction_dataset.csv` dataset, which contains demographic information, clinical measurements, lifestyle habits, and target risk indicators.

### Key Features:
* **Demographics**: `Age`, `Gender`, `Country`, `Work_Type`, `Residence_Type`
* **Physical & Clinical Metrics**: `Height_cm`, `Weight_kg`, `BMI`, `Blood_Pressure_Systolic`, `Blood_Pressure_Diastolic`, `Heart_Rate`, `Blood_Glucose`, `HbA1c`, `Total_Cholesterol`, `HDL`, `LDL`, `Triglycerides`
* **Lifestyle & Environment**: `Smoking_Status`, `Alcohol_Consumption`, `Physical_Activity_Level`, `Sleep_Hours`, `Stress_Level`, `Diet_Quality`, `Exercise_Hours_Per_Week`, `Daily_Walking_Minutes`, `Air_Pollution_Exposure`
* **Medical History**: `Family_History_Stroke`, `Family_History_Heart_Disease`, `Diabetes`, `Hypertension`, `Heart_Disease`, `Previous_TIA`, `Atrial_Fibrillation`, `Chronic_Kidney_Disease`, `Medication_Adherence`
* **Target Variable**: `Stroke_Risk` (`Low`, `Moderate`, `High`)

---

## 🛠️ Data Preprocessing & Pipeline Steps

1. **Missing Value Imputation**:
   * Continuous variables (`Age`, `Height_cm`, `Weight_kg`, `Blood_Glucose`, `HbA1c`, `Total_Cholesterol`, `HDL`, `LDL`, `Triglycerides`, `Sleep_Hours`, `Exercise_Hours_Per_Week`, `Daily_Walking_Minutes`) are imputed using column means.
   * Categorical variables (`Medication_Adherence`) are imputed using the mode.

2. **Feature & Target Separation**:
   * Redundant identifiers (`Patient_ID`) and metadata scores (`Stroke_Risk_Score`, `AI_Health_Recommendation`, `Doctor_Consultation_Needed`) are dropped from predictor features ($X$).
   * $y$ is defined using the target `Stroke_Risk`.

3. **Categorical Encoding**:
   * One-Hot Encoding (`pd.get_dummies`) with `drop_first=True` is applied to convert categorical features into binary numerical indicators.

4. **Train-Test Stratification**:
   * Data is split into training (70%) and testing (30%) sets using stratified sampling (`stratify=y`) to maintain class balance across target levels (`Moderate`, `High`, `Low`).

---

## 🤖 Model Evaluation & Results

A **Multinomial Logistic Regression** model was trained and evaluated on 15,000 test samples.

### Confusion Matrix
```text
[[ 3318     0   316 ]   # High
 [    0   874   180 ]   # Low
 [ 1595   814  7903 ]]  # Moderate

```

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | --- | --- | --- | --- |
| **High** | 0.68 | 0.91 | 0.78 | 3,634 |
| **Low** | 0.52 | 0.83 | 0.64 | 1,054 |
| **Moderate** | 0.94 | 0.77 | 0.84 | 10,312 |

### Overall Performance Metrics

* **Accuracy**: ~80.63% (81%)
* **Macro Precision**: 71%
* **Macro Recall**: 84%
* **Macro F1-Score**: 75%
* **Weighted F1-Score**: 81%

> **Key Takeaway**: The balanced Logistic Regression model successfully identifies high-risk individuals with a high sensitivity/recall rate of **91%**, which is critical for early medical intervention and screening applications.

---

## 🛠️ Tech Stack & Requirements

* **Python 3.8+**
* **pandas**
* **scikit-learn**
* **Jupyter Notebook**


---

## 🚀 How to Run

1. Place `stroke_risk_prediction_dataset.csv` in the root directory.
2. Launch the Jupyter Notebook interface:

```bash
jupyter notebook

```

3. Open `stroke risk prediction Multinomial Logistic Regression.ipynb` and run all cells.

---

## 📈 Future Enhancements

* Experiment with tree-based models (e.g., Random Forest, XGBoost, LightGBM) to capture non-linear relationships.
* Implement feature scaling using `StandardScaler` to optimize numerical feature weighting.
* Explore SMOTE (Synthetic Minority Over-sampling Technique) to handle class imbalances more effectively.

```

```
