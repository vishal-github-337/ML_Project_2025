# Heart Disease Risk Assesment

# a. Problem Statement
Provide a clinical-screening tool to identify individuals at risk of heart disease from routine health features. The primary objective is to maximize sensitivity (RECALL) to minimize missed positive cases, while keeping secondary attention on specificity and precision to limit unnecessary follow-ups.

# b. Dataset Description
The dataset model/heart_disease_risk_dataset_earlymed.csv contains patient records(70000, 19 ) for heart disease risk prediction.

- **Features**: Chest_Pain, Shortness_of_Breath, Fatigue, Palpitations, Dizziness, Swelling, Pain_Arms_Jaw_Back, Cold_Sweats_Nausea, High_BP, High_Cholesterol, Diabetes, Smoking, Obesity, Sedentary_Lifestyle, Family_History, Chronic_Stress, Gender, Age.

- **Target**: Heart_Risk (Binary: 0 = Healthy, 1 = At Risk).

## c. Models Used and Comparison

We implemented 6 classification models. The evaluation metrics on the test set are summarized below:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9911 | 0.9995 | 0.9913 | 0.9909 | 0.9911 | 0.9821 |
| **Decision Tree** | 0.9818 | 0.9818 | 0.9813 | 0.9823 | 0.9818 | 0.0936 |
| **kNN** | 0.9923 | 0.9976 | 0.9924 | 0.9921 | 0.9923 | 0.9846 |
| **Naive Bayes** | 0.9911 | 0.9995 | 0.9913 | 0.9909 | 0.9911 | 0.9821 |
| **Random Forest** | 0.9916 | 0.9995 | 0.9920 | 0.9913 | 0.9916 | 0.9833 |
| **XGBoost** | 0.9936 | 0.9997 | 0.9939 | 0.9933 | 0.9936 | 0.9871 |

