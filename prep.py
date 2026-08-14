import numpy as np
import pandas as pd

df = pd.read_csv("./airbnb_described.csv")
df.columns
df2 = df.drop(
    columns=[
        "index",
        "latitude",
        "longitude",
        # "neighbourhood_group",
        # "last_review",
        # "host_id",
        # "host_name",
        # "n_pois_50m",
        # "latitude",
        # "longitude",
        # "neighbourhood",
        # "license",
    ]
)
df2.to_csv("df2.csv", index=False)
