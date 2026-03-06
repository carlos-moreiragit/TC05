from datetime import timedelta
from feast import Entity, FeatureView, FileSource, Field, FeatureStore
from feast.types import Float32, Int64
from datetime import datetime
from zoneinfo import ZoneInfo

pk_entity = Entity(
    name="PK",
    join_keys=["PK"]
)

source = FileSource(
    #path="data/PEDE2024-silver.csv",
    path="data/parquet/PEDE2024-silver.parquet",
    timestamp_field="event_timestamp",
)

aluno_features = FeatureView(
    name="aluno_features",
    entities=[pk_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="INDE", dtype=Float32),
        Field(name="IDADE", dtype=Int64),
        Field(name="IAA", dtype=Float32),
        Field(name="IEG", dtype=Float32),
        Field(name="IPS", dtype=Float32),
        Field(name="IPP", dtype=Float32),
        Field(name="IDA", dtype=Float32),
        Field(name="IPV", dtype=Float32),
        Field(name="DEFASAGEM", dtype=Float32)
    ],
    source=source,
)

if __name__ == "__main__":
    store = FeatureStore(repo_path=".")
    store.apply([pk_entity, aluno_features])
    store.materialize_incremental(end_date=datetime(2030, 1, 1, 0, 0, 0, 0, tzinfo=ZoneInfo("America/Sao_Paulo")))