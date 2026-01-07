import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load dataset from CSV
df = pd.read_csv("train.csv")

# Select features
df = df[[
    "Survived",
    "Pclass",
    "Sex",
    "Age",
    "Fare",
    "Embarked"
]]

# Handle missing values
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)

# Encode categorical variables
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "titanic_model.pkl")

print("✅ Model trained using train.csv and saved as titanic_model.pkl")
