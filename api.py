from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from pathlib import Path

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train_mean_sample.csv"
TEST_PATH = BASE_DIR / "test_mean_sample.csv"
MODEL_PATH = BASE_DIR / "models" / "final_pipeline.joblib"
THRESHOLD_PATH = BASE_DIR / "models" / "final_threshold.joblib"

# =========================
# Chargement des données
# =========================
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)

train_data["client_id"] = range(1, len(train_data) + 1)
test_data["client_id"] = range(1, len(test_data) + 1)

FEATURE_COLUMNS = [
    col for col in train_data.columns
    if col not in ["TARGET", "client_id"]
]

X_train = train_data[FEATURE_COLUMNS].copy()

# =========================
# Chargement du modèle
# =========================
pipeline = joblib.load(MODEL_PATH)

if THRESHOLD_PATH.exists():
    threshold = joblib.load(THRESHOLD_PATH)
else:
    threshold = 0.2

# Récupération du modèle final si pipeline sklearn
try:
    model = pipeline.named_steps["model"]
except Exception:
    model = pipeline

# =========================
# Importance globale
# =========================
try:
    if hasattr(model, "feature_importances_"):
        global_shap_importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
    else:
        global_shap_importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": np.zeros(len(FEATURE_COLUMNS))
        })
except Exception:
    global_shap_importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": np.zeros(len(FEATURE_COLUMNS))
    })


# =========================
# Fonctions utilitaires
# =========================
def get_client_features(client_id):
    client_data = test_data[test_data["client_id"] == client_id]

    if client_data.empty:
        return None

    client_features = client_data[FEATURE_COLUMNS].copy()
    return client_features


def predict_client(client_features):
    proba = float(pipeline.predict_proba(client_features)[0][1])
    prediction_class = int(proba >= threshold)

    decision = "Crédit refusé" if prediction_class == 1 else "Crédit accordé"

    return {
        "prediction": proba,
        "prediction_proba": proba,
        "threshold": float(threshold),
        "prediction_class": prediction_class,
        "decision": decision
    }


# =========================
# Routes API
# =========================
@app.route("/")
def home():
    return "API de scoring crédit - Prêt à dépenser"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": MODEL_PATH.exists(),
        "threshold": float(threshold),
        "n_clients_test": len(test_data),
        "n_features": len(FEATURE_COLUMNS)
    })


@app.route("/check_client/<int:client_id>", methods=["GET"])
def check_client_id(client_id):
    exists = client_id in list(test_data["client_id"])
    return jsonify(exists)


@app.route("/client_info/<int:client_id>", methods=["GET"])
def get_client_info(client_id):
    client_data = test_data[test_data["client_id"] == client_id]

    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    return jsonify(client_data.to_dict(orient="records")[0])


@app.route("/client_info/<int:client_id>", methods=["PUT"])
def update_client_info(client_id):
    global test_data

    data = request.get_json()

    client_data = test_data[test_data["client_id"] == client_id]

    if client_data.empty:
        return jsonify({"error": "Client not found"}), 404

    for key, value in data.items():
        if key in test_data.columns and key != "client_id":
            test_data.loc[test_data["client_id"] == client_id, key] = value

    return jsonify({"message": "Client information updated"}), 200


@app.route("/client_info", methods=["POST"])
def submit_new_client():
    global test_data

    data = request.get_json()

    new_client_id = len(test_data) + 1
    data["client_id"] = new_client_id

    new_client = pd.DataFrame(data, index=[0])

    for col in test_data.columns:
        if col not in new_client.columns:
            new_client[col] = np.nan

    new_client = new_client[test_data.columns]

    test_data = pd.concat([test_data, new_client], ignore_index=True)

    return jsonify({
        "message": "New client submitted",
        "client_id": new_client_id
    }), 201


@app.route("/prediction", methods=["POST"])
def get_prediction():
    data = request.get_json()
    client_id = data.get("client_id")

    if client_id is None:
        return jsonify({"error": "client_id is required"}), 400

    client_features = get_client_features(client_id)

    if client_features is None:
        return jsonify({"error": "Client not found"}), 404

    try:
        result = predict_client(client_features)
        return jsonify(result)

    except Exception as e:
        return jsonify({
            "error": "Prediction failed",
            "details": str(e)
        }), 500


@app.route("/global_feature_importance", methods=["GET"])
def global_feature_importance():
    top_10 = global_shap_importance.head(10)
    result = top_10.set_index("feature")["importance"].to_dict()
    return jsonify(result)


@app.route("/local_feature_importance/<int:client_id>", methods=["GET"])
def local_feature_importance(client_id):
    client_features = get_client_features(client_id)

    if client_features is None:
        return jsonify({"error": "Client not found"}), 404

    try:
        # Cas simple : modèle LightGBM directement exploitable
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(client_features, check_additivity=False)

        values = np.abs(shap_values.values[0])

        local_importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": values
        }).sort_values("importance", ascending=False)

    except Exception:
        # Fallback simple si SHAP échoue
        if hasattr(model, "feature_importances_"):
            values = np.abs(model.feature_importances_)
        else:
            values = np.zeros(len(FEATURE_COLUMNS))

        local_importance = pd.DataFrame({
            "feature": FEATURE_COLUMNS,
            "importance": values
        }).sort_values("importance", ascending=False)

    result = local_importance.head(10).set_index("feature")["importance"].to_dict()
    return jsonify(result)


@app.route("/shap_summary_plot/<int:client_id>", methods=["GET"])
def shap_summary_plot(client_id):
    client_features = get_client_features(client_id)

    if client_features is None:
        return jsonify({"error": "Client not found"}), 404

    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(client_features, check_additivity=False)

        shap.summary_plot(
            shap_values.values,
            client_features,
            plot_type="bar",
            max_display=10,
            show=False
        )

        buf = BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        plt.close()
        buf.seek(0)

        image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return jsonify({"shap_summary_plot": image_base64})

    except Exception as e:
        return jsonify({
            "error": "SHAP plot failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)


