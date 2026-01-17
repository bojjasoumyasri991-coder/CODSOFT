# Titanic Survival Prediction
# CodSoft Data Science Internship - Task 1

# -----------------------------
# 1. Import required libraries
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------
# 2. Load the dataset
# -----------------------------
data = pd.read_csv(
    r"C:\Users\sujit\Downloads\archive\Titanic-Dataset.csv"
)
print("First 5 rows of the dataset:")
print(data.head())

# -----------------------------
# 3. Dataset information
# -----------------------------
print("\nDataset Info:")
print(data.info())

# -----------------------------
# 4. Data Cleaning
# -----------------------------

# Fill missing Age values with median
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# -----------------------------
# 5. Encode categorical columns
# -----------------------------
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# -----------------------------
# 6. Feature selection
# -----------------------------
X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked']]
y = data['Survived']

# -----------------------------
# 7. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# 8. Train Logistic Regression model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 9. Prediction and Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\nModel Accuracy:", accuracy)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nTask 1 completed successfully!")
