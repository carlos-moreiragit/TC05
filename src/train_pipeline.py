import pandas as pd
from feast import FeatureStore
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

store = FeatureStore(repo_path=".")

df = pd.read_csv("data/PEDE2024-silver.csv", parse_dates=["event_timestamp"])

entity_df = df[["PK", "event_timestamp"]]

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "aluno_features:INDE",
        "aluno_features:IDADE",
        "aluno_features:IAA",
        "aluno_features:IEG",
        "aluno_features:IPS",
        "aluno_features:IPP",
        "aluno_features:IDA",
        "aluno_features:IPV",
    ],
).to_df()

training_df["DEFASAGEM"] = df["DEFASAGEM"].values

X = training_df.drop(columns=["PK", "event_timestamp", "DEFASAGEM"])
y = training_df["DEFASAGEM"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True, class_weight = "balanced"))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "modelo_svm_defasagem.pkl")
print("Modelo salvo em modelo_svm_defasagem.pkl")