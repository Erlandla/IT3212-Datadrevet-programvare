from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

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

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

# Instantiate model with 1000 decision trees 
rf = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=50, min_samples_split=2, min_samples_leaf=1)
# Train the model on training data
rf.fit(train_features, train_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
