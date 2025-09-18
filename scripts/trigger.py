import json, sys, pandas as pd
from datetime import datetime
from sklearn.linear_model import LogisticsRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from joblib import load
from scipy.stats import ks_2samp

with open("state.json") as file:
    state = json.load(file)

modelPath = state["currentModel"]
ref_csv = state["features"]
minPerf = state["minPerf"]
pThres = state["pThreshold"]

classifier = load(modelPath)

X_ref = pd.read_csv(ref_csv)
X_new = pd.read_csv("data/new.csv")
y_new = X_new.pop("target") if "target" in X_new.columns else None

X_new = X_new[X_ref.columns] #aligning cols

perfDrop = False
f1 = None
if y_new is not None:
    preds = classifier.predict(X_new)
    f1 = f1_score(y_new, preds)
    perfDrop = f1 < minPerf

driftFlags = []
for col in X_ref.columns:
    try:
        stat, p = ks_2samp(X_ref[col].astype(float), X_new[col].astype(float), alternative="two-sided", mode="auto")
        driftFlags.append(p < pThres)
    except Exception:
        pass
drift = any(driftFlags)

#checking cuz im scared

print({"f1": f1, "perf_drop": perf_drop, "drift": drift})

if perfDrop or drift:
    print("Restrain Needed")
    sys.exit(1)
else:
    print("No Retrain")
    sys.exit(0)