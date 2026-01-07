import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = sns.load_dataset("titanic")

# Select features
df = df[["survived", "pclass", "sex", "age", "fare", "embarked"]]
df.dropna(inplace=True)

# Encode categorical variables
df["sex"] = df["sex"].map({"male": 0, "female": 1})
df["embarked"] = df["embarked"].map({"S": 0, "C": 1, "Q": 2})

X = df.drop("survived", axis=1)
y = df["survived"]

# Train model
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, "titanic_model.pkl")

print("Model trained and saved!")
