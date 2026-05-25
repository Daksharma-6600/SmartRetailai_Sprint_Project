import pandas as pd
from sklearn.ensemble import IsolationForest

import pandas as pd
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "train.csv"
)

df = pd.read_csv(DATA_PATH)

def detect_anomalies():

    data = df[["Sales"]]

    model = IsolationForest(
        contamination=0.02,
        random_state=42
    )

    df["Anomaly"] = model.fit_predict(data)

    anomalies = df[df["Anomaly"] == -1]

    return {
        "total_anomalies": len(anomalies),
        "highest_anomaly_sales": float(anomalies["Sales"].max()),
        "average_anomaly_sales": float(
            anomalies["Sales"].mean()
        )
    }