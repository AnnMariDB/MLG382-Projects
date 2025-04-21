# src/web_app.py

import os
import dash
from dash import html, dcc, Input, Output, State
import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model
import plotly.express as px

# Absolute path setup
base_dir = os.path.dirname(os.path.abspath(__file__))
artifacts_dir = os.path.join(base_dir, "..", "artifacts")

# Label map
grade_labels = {
    0: "Fail",
    1: "Pass",
    2: "Good",
    3: "Very Good",
    4: "Excellent"
}

# Load models and artifacts
model_1 = joblib.load(f"{artifacts_dir}/model_1.pkl")
model_2 = joblib.load(f"{artifacts_dir}/model_2.pkl")
model_3 = joblib.load(f"{artifacts_dir}/model_3.pkl")
model_4 = load_model(f"{artifacts_dir}/model_4.keras")
scaler = joblib.load(f"{artifacts_dir}/scaler.pkl")
X_columns = joblib.load(f"{artifacts_dir}/X_columns.pkl")

# Load predictions.csv for accuracy comparison
preds_df = pd.read_csv(f"{artifacts_dir}/predictions.csv")

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Student Grade Predictor"

# App layout
app.layout = html.Div([
    html.H1("🎓 Student Grade Prediction Dashboard", style={"textAlign": "center", "color": "#2c3e50"}),

    dcc.Tabs([
        # Visualization Tab
        dcc.Tab(label="Visualizations", children=[
            html.Br(),
            html.H3("Model Accuracy Comparison", style={"textAlign": "center"}),
            dcc.Graph(
                figure=px.bar(
                    preds_df.drop(columns="Actual").apply(lambda col: (preds_df["Actual"] == col).mean(), axis=0).reset_index(),
                    x="index", y=0,
                    labels={"index": "Model", "0": "Accuracy"},
                    color="index",
                    title="Model Accuracy",
                    template="plotly_white"
                )
            ),
            html.Br(),
            html.H4("Confusion Matrix (Deep Learning)", style={"textAlign": "center"}),
            html.Img(src="/assets/cm_dl.png", style={"maxWidth": "70%", "margin": "0 auto", "display": "block"})
        ]),

        # Prediction Tab
        dcc.Tab(label=" Predict Grades", children=[
            html.Div([
                html.P("Enter Feature Values:", style={"fontWeight": "bold"}),
                html.Div([
                    html.Div([
                        html.Label(col),
                        dcc.Input(id=col, type="number", placeholder=col, value=0, style={"width": "100%"})
                    ], style={"margin": "5px", "flex": "1"}) for col in X_columns
                ], style={"display": "flex", "flexWrap": "wrap"}),

                html.Button("Predict Grade", id="predict-btn", style={"marginTop": "15px"}),
                html.Div(id="prediction-output", style={
                    "marginTop": "25px",
                    "fontSize": "18px",
                    "background": "#f9f9f9",
                    "padding": "15px",
                    "borderRadius": "8px",
                    "boxShadow": "0 2px 5px rgba(0, 0, 0, 0.1)"
                })
            ], style={"padding": "20px"})
        ])
    ])
], style={"padding": "20px"})


@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State(col, "value") for col in X_columns]
)
def predict_grade(n_clicks, *input_values):
    if n_clicks is None:
        return ""
    
    input_array = np.array(input_values).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    preds = {
        "Logistic Regression": grade_labels[model_1.predict(input_scaled)[0]],
        "Random Forest": grade_labels[model_2.predict(input_array)[0]],
        "XGBoost": grade_labels[model_3.predict(input_array)[0]],
        "Deep Learning": grade_labels[np.argmax(model_4.predict(input_scaled)[0])]
    }

    return html.Ul([
        html.Li(f"{model}: {grade}", style={"marginBottom": "8px"}) for model, grade in preds.items()
    ])

if __name__ == "__main__":
    app.run(debug=True)
