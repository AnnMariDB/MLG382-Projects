# src/train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import joblib
import os

# Load your data
df = pd.read_csv("../Student_performance_data.csv")

# Feature and target
X = df.drop(columns=["Grade"])
y = df["Grade"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
artifacts_dir = os.path.join("..", "MLG382Project1", "artifacts")
os.makedirs(artifacts_dir, exist_ok=True)
joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

# Train and save models
model_1 = LogisticRegression(max_iter=1000)
model_1.fit(X_train_scaled, y_train)
joblib.dump(model_1, os.path.join(artifacts_dir, "model_1.pkl"))

model_2 = RandomForestClassifier(random_state=42)
model_2.fit(X_train, y_train)  # unscaled
joblib.dump(model_2, os.path.join(artifacts_dir, "model_2.pkl"))

model_3 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model_3.fit(X_train, y_train)
joblib.dump(model_3, os.path.join(artifacts_dir, "model_3.pkl"))

# Optional: Save predictions
pred_df = pd.DataFrame({
    "Actual": y_test,
    "Logistic Regression": model_1.predict(X_test_scaled),
    "Random Forest": model_2.predict(X_test),
    "XGBoost": model_3.predict(X_test),
})

pred_df.to_csv(os.path.join(artifacts_dir, "predictions.csv"), index=False)