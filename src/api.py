from flask import Flask, request, jsonify
from feast import FeatureStore
import joblib
import pandas as pd
import logging
from datetime import datetime

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("predictions.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

model = joblib.load("../model/passos.pkl")
store = FeatureStore(repo_path=".")

FEATURES = [
    "aluno_features:INDE",
    "aluno_features:IDADE",
    "aluno_features:IAA",
    "aluno_features:IEG",
    "aluno_features:IPS",
    "aluno_features:IPP",
    "aluno_features:IDA",
    "aluno_features:IPV",
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    ra = data.get("RA")
    ano = data.get("ANO")

    if ra is None or ano is None:
        logger.warning("Requisição inválida: %s", data)
        return jsonify({"error": "Campos RA e ANO são obrigatórios"}), 400

    entity_rows = [{"RA": ra, "ANO": ano}]

    features = store.get_online_features(
        features=FEATURES,
        entity_rows=entity_rows,
    ).to_df()

    X = features.drop(columns=["RA", "ANO"])

    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "RA": ra,
        "ANO": ano,
        "classe_defasagem": pred,
        "probabilidade": prob
    }

    logger.info(f"PREDICTION | {log_data}")

    return jsonify(log_data)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)