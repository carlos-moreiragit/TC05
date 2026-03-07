from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import time

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("../logs/predictions.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

model = joblib.load("../model/passos.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    start_time = time.perf_counter()

    # data recebe JSON do request body
    data = request.get_json(force=True)

    input_data = pd.DataFrame(data)

    prediction = model.predict(input_data) # Reshape para single prediction

    prediction_list = prediction.tolist()

    end_time = time.perf_counter()

    logging.info(f"Tempo de processamento: {end_time - start_time:.4f} segundos")

    # retorna JSON
    return jsonify({'prediction': prediction_list})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)