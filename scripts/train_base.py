import json, os, pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticsRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from joblib import dump

df = pd.read_csv("data/train.csv")
targetedColumn = "target"

X, y = df.drop(columns=[targetedColumn]), df[targetedColumn]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.30, random_state=0, stratify=y)

classifier = LogisticsRegression(maxIterations = 100)
classifier.fit(X_train, y_train)

f1 = f1_score(y_val, classifier.predict(X_val))
os.makedirs("models", exist_ok=True)
dump(classifier, "models/current.pkl")

os.makedirs("reports", exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
pd.DataFrame([{"timestamp": timestamp, "split": "val", "f1": f1}]).to_csv(f"reports/reports.csv", index=False)

statePath = "state.json"
if not os.path.exists(statePath):
    with open(statePath, "w") as f:
        json.dump({
            "metricName": "f1",
            "minPerf": 0.82,
            "driftTest": "ks",
            "pThreshold": 0.01,
            "currentModel": "models/current.pkl",
            "features": "data/ref.csv"
        }), f, indent=2