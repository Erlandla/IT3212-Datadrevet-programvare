import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("Tetuan City power consumption.csv")

# Set index as datetimes
data.set_index("DateTime", inplace=True)
data.index = pd.to_datetime(data.index)

data_copy = data.copy()


zone1 = data_copy["Zone 1 Power Consumption"]
zone2 = data_copy["Zone 2  Power Consumption"]
zone3 = data_copy["Zone 3  Power Consumption"]


def moving_average(x, D):
    """
    Simple moving average taken from the slides of IT3212 Datadrevet programvare.

    Takes a column-dataframe `x` and a window size `D` and smooths out the values by taking the 
    average of every value datapoint within that window

    ### Params
    x: column, preferrably a DataFrame
    D: window size

    ### Returns
    arraylike consisting of the moved average values of the given column
    """
    h = [v for v in x[:D]]
    for p in range(len(x) - D):
        h.append(sum(x[p:p+D])/float(D))
    return h


def visualise_zone1_smoothed():
    figure, axis = plt.subplots(2, 1)

    # Important resource for determining window size: https://medium.com/@thedatabeast/time-series-part-4-determining-the-window-size-for-moving-averages-a07c5cfcfac9
    # Some points from it:
    #   - Optimal window size based on trial and error. Using the validation set may help in this endeavour
    #   - Window size should be a multiple of the seasonal period to ensure that the seasonal pattern is captured
    #      properly
    #   - The window size should be chosen based on the characteristics of the time series and the forecasting
    #      problem at hand. There is no one-size-fits-all approach, and the optimal window size can vary depending
    #      on the data.
    #   - Longer window size: capture long-term trends
    #   - Small size: capture short-term fluctuations and changes
    #   - Ex.: If the time series has a yearly seasonality, a 12-month window size might be appropriate

    # --- Default ---
    axis[0].set_title("Zone 1 default consumption values")
    axis[0].plot(zone1)

    # Expect daily seasonality, so with this dataset it translates to 6*24 = 144 data points
    window_size = 144

    axis[1].set_title(
        "Zone 1 Consumption after moving average, window size = " + str(window_size))
    axis[1].plot(moving_average(zone1, window_size))

    plt.show()


if __name__ == "__main__":
    visualise_zone1_smoothed()
