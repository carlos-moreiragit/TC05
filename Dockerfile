FROM python:3.10-slim

WORKDIR /app

# copiar arquivos
COPY . /app

# instalar dependências
RUN pip install --no-cache-dir \
    flask \
    feast \
    pandas \
    scikit-learn \
    joblib

EXPOSE 5000

CMD ["python", "api.py"]