import pandas as pd
import numpy as np
from typo_correction import typo_correction
from matplotlib import pyplot as plt
from aggregate_zone_consumption import aggregate_zone_consumptions


def get_bands(data):
    return (np.mean(data) + 3 * np.std(data), np.mean(data) - 3 * np.std(data))

# get_bands = lambda data : (np.mean(data) + np.nanquantile(data, 0.95), np.mean(data) - np.nanquantile(data, 0.95))


def main():
    data = pd.read_csv("Tetuan City power consumption.csv")

    data = typo_correction(data)

    data = aggregate_zone_consumptions(data)

    column = data.loc[:, ("temperature")]

    print(data.columns)

    window_percentage = 3
    k = int(len(column) * (window_percentage/2/100))
    N = len(column)

    bands = [get_bands(column[range(0 if i-k < 0 else i-k, i+k if i+k < N else N)])
             for i in range(0, N)]
    upper, lower = zip(*bands)

    anomalies = (column > upper) | (column < lower)

    column.plot()

    count = 0
    for i in range(len(anomalies)):
        if (anomalies[i] == True):
            column[i] = None
            count += 1

    if count == 0:
        print("No anomalies detected")

    column.plot()

    column = column.interpolate(method="linear")

    plt.legend()
    plt.show()


def outlier_detection(df: pd.DataFrame, column: pd.Series):

    window_percentage = 3
    k = int(len(column) * (window_percentage/2/100))
    N = len(column)

    bands = [get_bands(column[range(0 if i-k < 0 else i-k, i+k if i+k < N else N)])
             for i in range(0, N)]
    upper, lower = zip(*bands)

    anomalies = (column > upper) | (column < lower)

    count = 0
    for i in range(len(anomalies)):
        if (anomalies[i] == True):
            print("ANOMALY on datetime " + df["datetime"][i] +
                  " with value " + str(column[i]))
            print("Removing anomaly...")
            column[i] = None
            count += 1

    if count == 0:
        print("No anomalies detected")

    column = column.interpolate(method="linear")

    return column


if __name__ == "__main__":
    main()
