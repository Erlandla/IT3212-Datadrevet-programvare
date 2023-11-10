import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import optuna.integration.lightgbm as lgb
from sklearn.preprocessing import StandardScaler, minmax_scale
import numpy as np
import optuna
from lightgbm import early_stopping
from lightgbm import log_evaluation
from sklearn.metrics import accuracy_score

df = pd.read_csv('./basic_models/preprocessed_data.csv')

df = df.rename(columns={'aggregated_consumption': 'consumption'})
df.drop(columns=["Unnamed: 0"], inplace=True, axis=1)
y = df["consumption"]


X = df.drop('consumption', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=42, stratify=None)
dtrain = lgb.Dataset(X_train, label=y_train)
dval= lgb.Dataset(X_test, label=y_test)

params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type:": "gbdt",
        "seed": 42

    }

model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=[early_stopping(100), log_evaluation(100)])

prediction = np.rint(model.predict(X_test, num_iteration=model.best_iteration))
accuracy = mean_absolute_error(y_test, prediction)

best_params = model.params
print("Best params:", best_params)
print("  MAE = {}".format(accuracy))
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))


