# Movie Rating Prediction
# CodSoft Data Science Internship - Task 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# -----------------------------
# Load dataset
# -----------------------------
data = pd.read_csv("IMDB-Dataset.csv")

print("First 5 rows of dataset:")
print(data.head())

print("\nDataset Info:")
print(data.info())

# -----------------------------
# Feature selection
# -----------------------------
X = data[['Year', 'Duration', 'Votes']]
y = data['Rating']

# -----------------------------
# Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train regression model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Prediction & evaluation
# -----------------------------
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R2 Score:", r2)

print("\nTask 2 completed successfully!")
