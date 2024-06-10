import requests
from io import BytesIO
from typing import List

import pandas as pd

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    months = ["03"]
    print("In")

    for i in months:
        data_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-{i}.parquet"
        df = pd.read_parquet(data_url)
        dfs.append(df)

    return pd.concat(dfs)