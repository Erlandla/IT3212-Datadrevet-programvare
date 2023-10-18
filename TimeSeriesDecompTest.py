import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot as plt

data = pd.read_csv("Tetuan City power consumption.csv")

data.set_index("DateTime", inplace=True)
data.index = pd.to_datetime(data.index)

# Drop null values
data.dropna(inplace=True)

result = seasonal_decompose(
    data["Zone 3  Power Consumption"], model="multiplicative", period=144)
result.seasonal.plot()
result.trend.plot()
fig = result.plot()
plt.show()
