import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import joblib
df = pd.read_csv('model/heart_disease_risk_dataset_earlymed.csv')

# Define features and target
X = df.drop('Heart_Risk', axis=1)
y = df['Heart_Risk']
 
# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
 
combined_df = pd.merge(X_test, y_test, left_index=True, right_index=True)
combined_df.to_csv("test.csv", index=False, encoding="utf-8")
 
combined_df = pd.merge(X_train, y_train, left_index=True, right_index=True)
combined_df.to_csv("train.csv", index=False, encoding="utf-8")
 
df = pd.read_csv("train.csv")

float_cols = df.select_dtypes(include=float).columns
df[float_cols] = df[float_cols].astype(int)

target_col = 'Heart_Risk'
# Prepare X and y
X_train = df.drop(target_col, axis=1)
y_train = df[target_col]
 
# Initialize Logistic Regression model
log_reg = LogisticRegression(max_iter=1000, random_state=42)
 
# Train the model
log_reg.fit(X_train, y_train)
 
# Save the model using joblib
filename = f'model/logistic_regression.joblib'
joblib.dump(log_reg, filename)
print(f"Saved {filename}")
 
 
# Initialize Decision Tree model
dt_classifier = DecisionTreeClassifier(random_state=42)
 
# Train the model
dt_classifier.fit(X_train, y_train)
 
# Save the model using joblib
 
filename = f'model/decision_tree.joblib'
joblib.dump(dt_classifier, filename)
print(f"Saved {filename}")
 
 
# Initialize K-Nearest Neighbors model
knn_classifier = KNeighborsClassifier(n_neighbors=5)
 
# Train the model
knn_classifier.fit(X_train, y_train)
 
# Save the model using joblib
 
filename = f'model/knn.joblib'
joblib.dump(knn_classifier, filename)
print(f"Saved {filename}")
 
 
# Initialize Naive Bayes model
nb_classifier = GaussianNB()
 
# Train the model
nb_classifier.fit(X_train, y_train)
 
# Save the model using joblib
 
filename = f'model/naive_bayes.joblib'
joblib.dump(nb_classifier, filename)
print(f"Saved {filename}")
 
 
# Initialize Random Forest model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
 
# Train the model
rf_classifier.fit(X_train, y_train)
 
# Save the model using joblib
 
filename = f'model/random_forest.joblib'
joblib.dump(rf_classifier, filename)
print(f"Saved {filename}")
 
 
# Initialize Gradient Boosting model
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
 
# Train the model
gb_classifier.fit(X_train, y_train)
 
# Save the model using joblib
 
filename = f'model/gradient_boosting.joblib'
joblib.dump(gb_classifier, filename)
print(f"Saved {filename}")
 
 
 
print("All models saved successfully.")