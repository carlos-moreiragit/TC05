import pickle
#import numpy as np
from flask import Flask, request, jsonify
import logging
import time
import pandas as pd


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Carrega o modelo globalmente
with open('../model/passos.pkl', 'rb') as f:
    model = pickle.load(f)

# Rota para fazer previsões
@app.route('/predict', methods=['POST'])
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

@app.route('/healthcheck', methods=['GET'])
def healthcheck():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
