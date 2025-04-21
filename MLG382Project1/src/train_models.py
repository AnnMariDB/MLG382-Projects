# src/train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

# Load raw data
df = pd.read_csv("../Student_performance_data.csv")

# Map numerical codes to strings for categorical variables
ethnicity_map = {
    0: "Caucasian",
    1: "AfricanAmerican",
    2: "Asian",
    3: "Other"
}

parentaleducation_map = {
    0: "None",
    1: "HighSchool",
    2: "SomeCollege",
    3: "Bachelors",
    4: "HigherStudy"
}

gender_map = {
    0: "Male",
    1: "Female"
}

df["ethnicity"] = df["ethnicity"].map(ethnicity_map)
df["parentaleducation"] = df["parentaleducation"].map(parentaleducation_map)
df["gender"] = df["gender"].map(gender_map)

# Convert boolean-like to integers (if not already)
for col in ["tutoring", "parental support", "extracurricular", "sports", "music", "volunteering"]:
    df[col] = df[col].astype(int)

# One-hot encode
categorical_cols = [
    "gender", "ethnicity", "parentaleducation",
    "tutoring", "parental support", "extracurricular", "sports", "music", "volunteering"
]

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Separate features and target
X = df_encoded.drop(columns=["Grade"])
y = df_encoded["Grade"]

# Save column order for prediction-time alignment
artifacts_dir = os.path.join("..", "MLG382Project1", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
joblib.dump(X.columns.tolist(), os.path.join(artifacts_dir, "X_columns.pkl"))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

# Train and save models
model_1 = LogisticRegression(max_iter=1000)
model_1.fit(X_train_scaled, y_train)
joblib.dump(model_1, os.path.join(artifacts_dir, "model_1.pkl"))

model_2 = RandomForestClassifier(random_state=42)
model_2.fit(X_train, y_train)
joblib.dump(model_2, os.path.join(artifacts_dir, "model_2.pkl"))

model_3 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_3.fit(X_train, y_train)
joblib.dump(model_3, os.path.join(artifacts_dir, "model_3.pkl"))

# Save predictions
pred_df = pd.DataFrame({
    "Actual": y_test,
    "Logistic Regression": model_1.predict(X_test_scaled),
    "Random Forest": model_2.predict(X_test),
    "XGBoost": model_3.predict(X_test),
})

pred_df.to_csv(os.path.join(artifacts_dir, "predictions.csv"), index=False)