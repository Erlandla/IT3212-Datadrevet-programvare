import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('Tetuan City power consumption.csv')


#column = data["Zone 1 Power Consumption"] # Column should look like this
def findOutliers(column):
    # Define a function to remove outliers from a given column
    window_percentage = 3
    k = int(len(column) * (window_percentage/2/100))
    N = len(column)

    get_bands = lambda data : (np.mean(data) + np.nanquantile(data,0.98),np.mean(data) - np.nanquantile(data,0.98))

    bands = [get_bands(column[range(0 if i - k < 0 else i-k ,i + k if i + k < N else N)]) for i in range(0,N)] 
    upper, lower = zip(*bands)

    anomalies = (column > upper) & (column < lower)

    #Returns list of all data points as booleans, True if the data point is an outlier
    return anomalies

# As of now, no anomalies are detected. In the eventuality that some are found, a method to replace them with e.g. the mean, should be implemented
# Should also document that this has been done, and no results from it has been achieved. 