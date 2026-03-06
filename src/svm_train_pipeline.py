import pandas as pd
from feast import FeatureStore
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
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
        "aluno_features:DEFASAGEM",
    ],
).to_df()

#training_df["DEFASAGEM"] = df["DEFASAGEM"].values

X = training_df.drop(columns=["PK", "event_timestamp", "DEFASAGEM"], axis = 1)
y = training_df["DEFASAGEM"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)

numeric_features = [col for col in X_train.columns if X_train[col].dtypes in ["int64", "float64"]]
categorical_features = [col for col in X_train.columns if X_train[col].dtypes in ["object"]]

cv = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)

def cross_val_report(pipe, X_train, y_train):
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train, y_train),1):
        X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
        pipe.fit(X_train_fold, y_train_fold)
        y_val_pred = pipe.predict(X_valid_fold)
        print(f"\nClassification report for fold {fold}:")
        print(classification_report(y_valid_fold, y_val_pred))
        report = classification_report(y_valid_fold, y_val_pred, output_dict=True)
        precision_scores.append(report['weighted avg']['precision'])
        recall_scores.append(report['weighted avg']['recall'])
        f1_scores.append(report['weighted avg']['f1-score'])

    print("\nMédia Ponderada da Acurácia: ",np.mean(precision_scores))
    print("Média Ponderada do recall: ",np.mean(recall_scores))
    print("Média Ponderada f1-score: ",np.mean(f1_scores))

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(missing_values = 0, strategy = "median")),
    ("scaler", RobustScaler())
])

# Categorical pipeline
catorical_transformer = Pipeline(steps=[
    ("imputer",SimpleImputer(strategy = "most_frequent")),
    ("encoder",OneHotEncoder(handle_unknown = "ignore"))
])

# ColumnTransformer
preprocessor_lr = ColumnTransformer([
    ("num", numeric_transformer,numeric_features),
    ("cat", catorical_transformer,categorical_features)
])

#preprocessor for tree-based model
preprocessor_treebased = ColumnTransformer([
    ("num", SimpleImputer(missing_values = 0, strategy = "median"), numeric_features),
    ("cat", OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value= -1), categorical_features)
])

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor_lr),
    ("classifier", SVC(kernel = 'rbf', probability = True, class_weight = "balanced"))
])

#pipeline.fit(X_train, y_train)

cross_val_report(pipeline,X_train,y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

joblib.dump(pipeline, "../model/passos.pkl")
print("Modelo salvo em model/passos.pkl")