import pandas as pd
from feast import FeatureStore

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


print(training_df.head())


