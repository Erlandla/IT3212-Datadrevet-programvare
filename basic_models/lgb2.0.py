import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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
df.drop(columns=df.columns[0], inplace=True, axis=1)
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
        "seed": 42,
        "max_depth": 5,
        "n_estimators": 1000,
    }

model = lgb.train(params, dtrain, valid_sets=[dval], callbacks=[early_stopping(100), log_evaluation(100)])

prediction = np.rint(model.predict(X_test, num_iteration=model.best_iteration))
mae = mean_absolute_error(y_test, prediction)

r2 = r2_score(y_test, prediction)
best_params = model.params
print("Best params:", best_params)
print("  MAE = {}".format(mae))
print(" R2 = {}".format(r2))
# Absolute errors, and mean absolute error
errors = abs(prediction - y_test)

mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print("Accuracy", round(accuracy, 4), '%.')
print("  Params: ")
for key, value in best_params.items():
    print("    {}: {}".format(key, value))

