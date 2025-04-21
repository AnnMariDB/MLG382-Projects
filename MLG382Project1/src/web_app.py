import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# === Handle paths safely ===
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

# === Load models & artifacts ===
model_1 = joblib.load(ARTIFACTS_DIR / "model_1.pkl")
model_2 = joblib.load(ARTIFACTS_DIR / "model_2.pkl")
model_3 = joblib.load(ARTIFACTS_DIR / "model_3.pkl")
dl_model = load_model(ARTIFACTS_DIR / "model_4.keras")
scaler = joblib.load(ARTIFACTS_DIR / "scaler.pkl")
preds_df = pd.read_csv(ARTIFACTS_DIR / "predictions.csv")

# === Load & preprocess training data for reference ===
df = pd.read_csv(DATA_DIR / "train.csv")
df.drop(columns=[col for col in df.columns if col.endswith("_label")], inplace=True)
df_encoded = pd.get_dummies(df.drop("GradeClass", axis=1))
X_columns = df_encoded.columns

# === Helper: Prepare input for models ===
def preprocess_input(data_dict):
    df_input = pd.DataFrame([data_dict])
    df_input = pd.get_dummies(df_input)
    for col in X_columns:
        if col not in df_input.columns:
            df_input[col] = 0
    df_input = df_input[X_columns]
    return df_input, scaler.transform(df_input)

# === Dash App Setup ===
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Student Grade Prediction App"

app.layout = dbc.Container([
    html.H1("Learner Performance Dashboard", className="text-center mt-4 mb-4"),

    dcc.Tabs([
        dcc.Tab(label="Model Evaluation", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="accuracy-bar"), width=6),
                dbc.Col(dcc.Graph(id="confusion-matrix"), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="gpa-vs-grade"), width=6),
                dbc.Col(dcc.Graph(id="study-vs-grade"), width=6),
            ])
        ]),
        dcc.Tab(label="Make a Prediction", children=[
            html.Br(),
            dbc.Row([
                dbc.Col([
                    html.H5("Enter Student Data"),
                    dbc.Input(id="age", placeholder="Age", type="number"),
                    dbc.Input(id="gpa", placeholder="GPA", type="number", step=0.01),
                    dbc.Input(id="studytime", placeholder="Study Time Weekly", type="number"),
                    dbc.Input(id="absences", placeholder="Absences", type="number"),
                    dbc.Select(id="gender", options=[
                        {"label": "Male", "value": 0},
                        {"label": "Female", "value": 1}
                    ], placeholder="Gender"),
                    dbc.Select(id="ethnicity", options=[
                        {"label": "White", "value": 0},
                        {"label": "Hispanic", "value": 1},
                        {"label": "Black", "value": 2},
                        {"label": "Asian", "value": 3}
                    ], placeholder="Ethnicity"),
                    dbc.Input(id="parentaledu", placeholder="Parental Education (0-5)", type="number"),
                    dbc.Checkbox(id="tutoring", label="Tutoring"),
                    dbc.Checkbox(id="support", label="Parental Support"),
                    dbc.Checkbox(id="extracurricular", label="Extracurricular"),
                    dbc.Checkbox(id="sports", label="Sports"),
                    dbc.Checkbox(id="music", label="Music"),
                    dbc.Checkbox(id="volunteering", label="Volunteering"),
                    html.Br(),
                    dbc.Button("Predict", id="predict-btn", color="primary"),
                ], width=4),
                dbc.Col([
                    html.Div(id="prediction-output", className="mt-4")
                ])
            ])
        ])
    ])
], fluid=True)

# === Update model evaluation graphs ===
@app.callback(
    Output("accuracy-bar", "figure"),
    Output("confusion-matrix", "figure"),
    Output("gpa-vs-grade", "figure"),
    Output("study-vs-grade", "figure"),
    Input("accuracy-bar", "id")
)
def update_graphs(_):
    accs = {
        "Logistic Regression": np.mean(preds_df["Actual"] == preds_df["Logistic Regression"]),
        "Random Forest": np.mean(preds_df["Actual"] == preds_df["Random Forest"]),
        "XGBoost": np.mean(preds_df["Actual"] == preds_df["XGBoost"]),
        "Deep Learning": np.mean(preds_df["Actual"] == preds_df["Deep Learning"]),
    }

    acc_fig = px.bar(x=list(accs.keys()), y=list(accs.values()), labels={"x": "Model", "y": "Accuracy"}, title="Model Accuracy")

    cm = pd.crosstab(preds_df["Actual"], preds_df["Deep Learning"], rownames=['Actual'], colnames=['Predicted'])
    cm_fig = px.imshow(cm, text_auto=True, title="Confusion Matrix - Deep Learning")

    gpa_fig = px.violin(df, x="GradeClass", y="GPA", box=True, points="all", title="GPA vs GradeClass")
    study_fig = px.scatter(df, x="StudyTimeWeekly", y="GPA", color="GradeClass", title="Study Time vs GPA")

    return acc_fig, cm_fig, gpa_fig, study_fig

# === Handle prediction request ===
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("age", "value"),
    State("gpa", "value"),
    State("studytime", "value"),
    State("absences", "value"),
    State("gender", "value"),
    State("ethnicity", "value"),
    State("parentaledu", "value"),
    State("tutoring", "value"),
    State("support", "value"),
    State("extracurricular", "value"),
    State("sports", "value"),
    State("music", "value"),
    State("volunteering", "value"),
)
def make_prediction(n_clicks, age, gpa, study, absences, gender, ethnicity, parentaledu,
                    tutoring, support, extracurricular, sports, music, volunteering):
    if not n_clicks:
        return ""
    
    # === Input validation ===
    required_fields = [age, gpa, study, absences, gender, ethnicity, parentaledu]
    if any(v is None for v in required_fields):
        return dbc.Alert("Please fill in all required fields before predicting.", color="warning")

    try:
        raw = {
            "Age": age,
            "GPA": gpa,
            "StudyTimeWeekly": study,
            "Absences": absences,
            "Gender": gender,
            "Ethnicity": ethnicity,
            "ParentalEducation": parentaledu,
            "Tutoring": int(bool(tutoring)),
            "Parental Support": int(bool(support)),
            "Extracurricular": int(bool(extracurricular)),
            "Sports": int(bool(sports)),
            "Music": int(bool(music)),
            "Volunteering": int(bool(volunteering))
        }

        input_encoded, input_scaled = preprocess_input(raw)

        preds = {
            "Logistic Regression": model_1.predict(input_scaled)[0],
            "Random Forest": model_2.predict(input_encoded)[0],
            "XGBoost": model_3.predict(input_encoded)[0],
            "Deep Learning": int(np.argmax(dl_model.predict(input_scaled), axis=1)[0])
        }

        return html.Div([
            html.H4("Predictions:"),
            html.Ul([html.Li(f"{model}: Grade {preds[model]}") for model in preds])
        ])

    except Exception as e:
        print("Prediction error:", str(e))
        return dbc.Alert(f"An error occurred during prediction: {str(e)}", color="danger")

# === Run the app ===
if __name__ == '__main__':
    app.run(debug=True)
