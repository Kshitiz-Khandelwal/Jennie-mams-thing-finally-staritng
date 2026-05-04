# =====================================
# HEART DISEASE PREDICTION - FLASK APP
# ML Model Comparison Dashboard
# =====================================

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score,
    precision_score, recall_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Store trained models and scaler globally for prediction
trained_models = {}
fitted_scaler = None
feature_columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def load_and_preprocess():
    """Load and preprocess the Cleveland heart disease dataset."""
    data = pd.read_csv("processed.cleveland.data", header=None)
    columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
    ]
    data.columns = columns
    data.replace("?", np.nan, inplace=True)
    data = data.apply(pd.to_numeric)
    data.fillna(data.median(), inplace=True)
    data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)
    return data


def train_and_evaluate():
    """Train all 3 models and return evaluation results."""
    global trained_models, fitted_scaler

    data = load_and_preprocess()
    X = data.drop("target", axis=1)
    y = data["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    fitted_scaler = scaler

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        cm = confusion_matrix(y_test, y_pred).tolist()
        report = classification_report(y_test, y_pred, output_dict=True)

        results[name] = {
            "accuracy": round(accuracy_score(y_test, y_pred) * 100, 2),
            "roc_auc": round(roc_auc_score(y_test, y_proba) * 100, 2),
            "precision": round(precision_score(y_test, y_pred) * 100, 2),
            "recall": round(recall_score(y_test, y_pred) * 100, 2),
            "f1": round(f1_score(y_test, y_pred) * 100, 2),
            "confusion_matrix": cm,
            "report": {
                "0": {
                    "precision": round(report["0"]["precision"] * 100, 2),
                    "recall": round(report["0"]["recall"] * 100, 2),
                    "f1-score": round(report["0"]["f1-score"] * 100, 2),
                    "support": int(report["0"]["support"])
                },
                "1": {
                    "precision": round(report["1"]["precision"] * 100, 2),
                    "recall": round(report["1"]["recall"] * 100, 2),
                    "f1-score": round(report["1"]["f1-score"] * 100, 2),
                    "support": int(report["1"]["support"])
                }
            }
        }
        trained_models[name] = model

    # Determine best model
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    results["best_model"] = best_name

    # Dataset info
    results["dataset_info"] = {
        "total_samples": len(data),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "features": len(feature_columns),
        "positive_class": int(y.sum()),
        "negative_class": int(len(y) - y.sum())
    }

    return results


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        results = train_and_evaluate()
        return jsonify({"status": "success", "data": results})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        if not trained_models:
            return jsonify({
                "status": "error",
                "message": "Models not trained yet. Please train first."
            }), 400

        input_data = request.json
        features = [float(input_data.get(col, 0)) for col in feature_columns]
        features_scaled = fitted_scaler.transform([features])

        predictions = {}
        for name, model in trained_models.items():
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            predictions[name] = {
                "prediction": "Heart Disease" if int(pred) == 1 else "No Heart Disease",
                "confidence": round(float(max(proba)) * 100, 2),
                "probability_disease": round(float(proba[1]) * 100, 2)
            }

        return jsonify({"status": "success", "data": predictions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
