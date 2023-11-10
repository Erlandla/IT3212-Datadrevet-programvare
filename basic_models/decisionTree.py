from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import time

# take the result from the preprocessing instead
df = pd.read_csv('./preprocessed_data.csv')

# Labels are the values we want to predict
labels = np.array(df['aggregated_consumption'])
# Remove the label from the features
df = df.drop('aggregated_consumption', axis=1)
# Convert to numpy array
features = np.array(df)

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42)

start_time = time.time()
# Instantiate Decision Tree model
dt = DecisionTreeRegressor(random_state=42)
# Train the model on training data
dt.fit(train_features, train_labels)
training_time = round(time.time() - start_time, 2)
print(f"Training time: {training_time} seconds")

# Use the model's predict method on the test data
predictions = dt.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (MAE)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
print(mape)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
