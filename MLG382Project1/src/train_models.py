# src/train_models.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def train_and_save_models(data_path, artifacts_dir):
    df = pd.read_csv(data_path)
    X = df.drop("gradeclass", axis=1)
    y = df["gradeclass"]

    # Save feature column names
    joblib.dump(list(X.columns), os.path.join(artifacts_dir, "X_columns.pkl"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(artifacts_dir, "scaler.pkl"))

    model_1 = LogisticRegression(max_iter=1000)
    model_1.fit(X_train_scaled, y_train)
    joblib.dump(model_1, os.path.join(artifacts_dir, "model_1.pkl"))

    model_2 = RandomForestClassifier(random_state=42)
    model_2.fit(X_train, y_train)
    joblib.dump(model_2, os.path.join(artifacts_dir, "model_2.pkl"))

    model_3 = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    model_3.fit(X_train, y_train)
    joblib.dump(model_3, os.path.join(artifacts_dir, "model_3.pkl"))

    num_classes = y.nunique()
    y_train_dl = to_categorical(y_train, num_classes=num_classes)

    dl_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    dl_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    dl_model.fit(X_train_scaled, y_train_dl, epochs=20, batch_size=32, validation_split=0.2, verbose=0)
    dl_model.save(os.path.join(artifacts_dir, "model_4.keras"))
