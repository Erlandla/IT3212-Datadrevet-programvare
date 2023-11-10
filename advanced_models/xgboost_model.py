import numpy as np
import pandas as pd
import time
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("./basic_models/preprocessed_data.csv")

# What we want to predict
target = np.array(df['aggregated_consumption'])

df = df.drop('aggregated_consumption', axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    df, target, test_size=0.25, random_state=42)

# +---- Hyperparameters ----+
"""
DEFINITIONS, from slide set #9:
eta (Default: 0.3): 
    - Learning rate
    - makes the model more robust by shrinking the 
        weights on each step
min_child_weight (Default: 1): 
    - minimum sum of weights of all observations required 
        in a child. 
    - Used to control overfitting. Higher values prevents 
        a model from learning relations which might be highly
        specific to the particular sample selected for a tree
max_depth (Default: 6): 
    - Defines maximum depth. 
    - Higher depth allows the model to learn relations very 
        specific to the particular sample.
gamma (Default: 0): 
    - A node is split only when the resulting split gives 
        a positive reduction in the loss function. Gamma
        specifies the minimum loss reduction required to make a split
    - Makes the algorithm conservative. The values can vary depending
        on the loss function and should be tuned.
subsample (Default: 1):
    - Same as the subsample of GBM. Denotes the fraction of observations 
        to be randomly sampled for each tree.
    - Lower values make the algorithm more conservative and prevent overfitting,
        but too small values might lead to underfitting
colsample_bytree (Default: 1):
    - It is similar to max_features in GBM
    - Denotes the fraction of columns to be randomly sampled for each tree
"""
# Default: 0.3
eta = 0.3
# Default: 1
min_child_weight = 1
# Default: 6
max_depth = 6
# Default: 0
gamma = 0
# Default: 1
subsample = 1
# Default: 1
colsample_bytree = 1

model = XGBRegressor(
    eta=eta, min_child_weight=min_child_weight,
    max_depth=max_depth,
    gamma=gamma, subsample=subsample,
    colsample_bytree=colsample_bytree,

)

"""
NOTES:
    - High training accuracy but low testing accuracy ==> overfitted
        - Can control overfitting by either directly controlling the 
            complexity (max_weight, min_child_weight, and gamma), or by
            adding randomness to make the model more robust to random noise
            (subsample, colsample_bytree) (can also reduce stepsize eta, but 
            remember to increase num_rounds when doing that)
"""

start_time = time.time()

model.fit(X_train, y_train)

training_time = (time.time() - start_time)
print("Execution time (seconds):", training_time)

predictions = model.predict(X_test)
print("Predictions", predictions)

print("\n+------ Importante stuff ------+")

# Absolute errors, and mean absolute error
errors = abs(predictions - y_test)
print("Mean Absolute Error:", round(np.mean(errors), 4))

# Accuracy, using Mean Avereage Percentage Error
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print("Accuracy", round(accuracy, 4), '%.')

# R^2-score
r_2 = model.score(X_test, y_test)
print("R-squared", round(r_2, 4))
