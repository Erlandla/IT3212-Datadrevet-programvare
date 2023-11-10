import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import optuna.integration.lightgbm as lgb
from sklearn.preprocessing import StandardScaler, minmax_scale
import numpy as np
import optuna

df = pd.read_csv('./basic_models/preprocessed_data.csv')

df = df.rename(columns={'aggregated_consumption': 'consumption'})
df.drop(columns=["Unnamed: 0"], inplace=True, axis=1)
y = df["consumption"]


X = df.drop('consumption', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False, random_state=42, stratify=None)


def objective(trial):
    params = {
        "objective": "regression",
        "metric": "mae",
        "n_estimators": 100,
        "random_state": 42,
        "max_depth": 20,
        "verbosity": -1,
        "bagging_freq": 1,
        "num_leaves": 4,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.3, 0.9),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 3, 8)

    }
    best_params, tuning_history = dict(), list()
    model = lgb.train(params, X_train, valid_sets=[X_test], best_params=best_params, tuning_history=tuning_history)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print("best_params", best_params)
    return mae

#Calculating mean absolute error
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

print("Best Hyperparameters:", study.best_params)
print("Best value (MAE):", study.best_value)
#Plotting real vs predicted values for a time period




'''
fig = plt.figure(figsize=(16,8))
plt.title(f'Real vs Prediction - MAE {mae}', fontsize=20)
plt.plot(y_test, color='red')
plt.plot(pd.Series(predictions, index=y_test.index), color='green')
plt.xlabel('Month', fontsize=16)
plt.ylabel('Consumption', fontsize=16)
plt.legend(labels=['Real', 'Prediction'], fontsize=16)
plt.grid()
plt.show()

errors = abs(predictions - y_test)
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print(round(accuracy, 2), '%.')
'''
