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
