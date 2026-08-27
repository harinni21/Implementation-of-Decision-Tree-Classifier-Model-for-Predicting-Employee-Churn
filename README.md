# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm 1: Decision Tree Classifier – Employee Churn
1.Upload and load the Employee.csv dataset and handle missing values.

2.Encode the categorical salary column and select the required features and left as the target.

3.Split the dataset into 80% training and 20% testing data.

4.Train a Decision Tree Classifier using the Entropy criterion and predict employee status.

5.Evaluate using accuracy, confusion matrix, and classification report, then visualize the decision tree.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Harinni S
RegisterNumber: 212224060093
# ============================================
# Employee Churn Prediction using Decision Tree
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------
# 1) Upload Employee.csv file
# --------------------------------------------

from google.colab import files

uploaded = files.upload()

# Get the uploaded file name
csv_path = list(uploaded.keys())[0]

# Load dataset
data = pd.read_csv(csv_path)

print("File uploaded successfully:", csv_path)

# --------------------------------------------
# 2) Quick inspection
# --------------------------------------------

print("\nData shape:", data.shape)

print("\nFirst 5 rows:")
display(data.head())

print("\nDataset information:")
data.info()

print("\nMissing values per column:")
print(data.isnull().sum())

print("\nTarget distribution (left):")
print(data["left"].value_counts())

# --------------------------------------------
# 3) Handle missing values
# --------------------------------------------

if data.isnull().any().any():
    print("\nDropping rows with missing values...")
    data = data.dropna()
    print("New shape after dropna:", data.shape)

# --------------------------------------------
# 4) Encode categorical column 'salary'
# --------------------------------------------

if "salary" in data.columns:

    le = LabelEncoder()

    data["salary"] = le.fit_transform(data["salary"].astype(str))

    print("\nSalary classes:")
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))

    print(mapping)

# --------------------------------------------
# 5) Define features and target
# --------------------------------------------

feature_cols = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "promotion_last_5years",
    "salary"
]

# Check whether all features exist
missing_feats = [c for c in feature_cols if c not in data.columns]

if missing_feats:
    raise ValueError(
        f"Missing expected feature columns: {missing_feats}"
    )

# Features
X = data[feature_cols]

# Target
y = data["left"]

print("\nFeatures:")
display(X.head())

print("\nTarget:")
display(y.head())

# --------------------------------------------
# 6) Train-test split
# --------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=100,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# --------------------------------------------
# 7) Create and train Decision Tree
# --------------------------------------------

dt = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)

dt.fit(X_train, y_train)

print("\nDecision Tree training completed.")

# --------------------------------------------
# 8) Prediction
# --------------------------------------------

y_pred = dt.predict(X_test)

# --------------------------------------------
# 9) Model Evaluation
# --------------------------------------------

accuracy = metrics.accuracy_score(y_test, y_pred)

print("\nAccuracy on test set:", round(accuracy, 4))

# Confusion Matrix
print("\nConfusion Matrix:")

cm = metrics.confusion_matrix(y_test, y_pred)

print(cm)

# Classification Report
print("\nClassification Report:")

print(
    metrics.classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

# --------------------------------------------
# 10) Predict for a new employee
# --------------------------------------------

# Format:
# [satisfaction_level,
#  last_evaluation,
#  number_project,
#  average_montly_hours,
#  time_spend_company,
#  Work_accident,
#  promotion_last_5years,
#  salary]

new_employee = [[
    0.5,    # satisfaction_level
    0.8,    # last_evaluation
    9,      # number_project
    260,    # average_montly_hours
    6,      # time_spend_company
    0,      # Work_accident
    1,      # promotion_last_5years
    2       # salary
]]

pred = dt.predict(new_employee)

print("\nNew Employee Data:")
print(new_employee[0])

if pred[0] == 1:
    print("Prediction: Employee will LEAVE")
else:
    print("Prediction: Employee will STAY")

# --------------------------------------------
# 11) Visualize Decision Tree
# --------------------------------------------

plt.figure(figsize=(18, 12))

plot_tree(
    dt,
    feature_names=feature_cols,
    class_names=["stayed", "left"],
    filled=True,
    rounded=True,
    fontsize=9
)

plt.title("Decision Tree for Employee Churn")

plt.show()

# --------------------------------------------
# 12) Feature Importance
# --------------------------------------------

importances = pd.Series(
    dt.feature_importances_,
    index=feature_cols
).sort_values(ascending=False)

print("\nFeature Importances:")

display(importances)  
*/
```

## Output:
<img width="1717" height="823" alt="image" src="https://github.com/user-attachments/assets/5f2c6bb1-a9cf-4b1c-9baf-bf07cb26050b" />

<img width="1535" height="778" alt="image" src="https://github.com/user-attachments/assets/a50b63b9-01ee-4257-9855-517c129c8e0d" />

<img width="1222" height="784" alt="image" src="https://github.com/user-attachments/assets/ef529f07-5bf0-418c-aee0-53b301e70a84" />

<img width="1359" height="103" alt="image" src="https://github.com/user-attachments/assets/e30d591b-4935-496d-a461-4ae3d061238c" />

<img width="1583" height="976" alt="image" src="https://github.com/user-attachments/assets/228ee5ad-0e47-4683-8998-0f854dc818a6" />

<img width="1111" height="443" alt="image" src="https://github.com/user-attachments/assets/254a469f-1f0b-406c-a559-b0ff6716054b" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
