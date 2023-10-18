import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def get_bands(data):
    return (np.mean(data) + 3 * np.std(data), np.mean(data) - 3 * np.std(data))

# get_bands = lambda data : (np.mean(data) + np.nanquantile(data, 0.95), np.mean(data) - np.nanquantile(data, 0.95))


def main():
    data = pd.read_csv("Tetuan City power consumption.csv")

    column = data["Zone 2  Power Consumption"]
    window_percentage = 3
    k = int(len(column) * (window_percentage/2/100))
    N = len(column)

    bands = [get_bands(column[range(0 if i-k < 0 else i-k, i+k if i+k < N else N)])
             for i in range(0, N)]
    upper, lower = zip(*bands)

    anomalies = (column > upper) | (column < lower)

    for i in range(len(anomalies)):
        if (anomalies[i] == True):
            print("ANOMALY on DateTime " + data["DateTime"][i] +
                  " with value " + str(data["Zone 2  Power Consumption"][i]))
            print("Removing anomaly...")
            column[i] = None

    column.plot()

    column = column.interpolate(method="linear")

    plt.legend()
    plt.show()


async def outlier_detection(df: pd.DataFrame, column: pd.Series):

    column = df["aggregated_consumption"]

    window_percentage = 3
    k = int(len(column) * (window_percentage/2/100))
    N = len(column)

    bands = [get_bands(column[range(0 if i-k < 0 else i-k, i+k if i+k < N else N)])
             for i in range(0, N)]
    upper, lower = zip(*bands)

    anomalies = (column > upper) | (column < lower)

    for i in range(len(anomalies)):
        if (anomalies[i] == True):
            print("ANOMALY on datetime " + df["datetime"][i] +
                  " with value " + str(column[i]))
            print("Removing anomaly...")
            column[i] = None

    # column = column.interpolate(method="linear")


if __name__ == "__main__":
    main()
