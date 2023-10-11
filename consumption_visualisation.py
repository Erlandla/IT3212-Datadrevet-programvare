import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("Tetuan City power consumption.csv")

"""
print("Consumption: Zone 1")
plt.title("Power consumption of Zone 1")
plt.xlabel("DateTime")
plt.xticks(np.arange(0, len(data["DateTime"])-1, (len(data["DateTime"])-1)/12))
plt.ylabel("Power consumption")
plt.plot(data["DateTime"], data["Zone 1 Power Consumption"])
plt.show()

print("Consumption: Zone 2")
plt.title("Power consumption of Zone 2")
plt.xlabel("DateTime")
plt.xticks(np.arange(0, len(data["DateTime"])-1, (len(data["DateTime"])-1)/12))
plt.ylabel("Power consumption")
plt.plot(data["DateTime"], data["Zone 2  Power Consumption"])
plt.show()

print("Consumption: Zone 3")
plt.title("Power consumption of Zone 3")
plt.xlabel("DateTime")
plt.xticks(np.arange(0, len(data["DateTime"])-1, (len(data["DateTime"])-1)/12))
plt.ylabel("Power consumption (kWh)")
plt.plot(data["DateTime"], data["Zone 3  Power Consumption"])
plt.show()
"""

print("Consumption: all zones together")
plt.title("Consumption: all zones together\nZ1 = green, Z2 = red, Z3 = blue")
plt.xlabel("DateTime")
plt.xticks(np.arange(0, len(data["DateTime"])-1, (len(data["DateTime"])-1)/12))
plt.ylabel("Power consumption (kWh)")
plt.plot(data["DateTime"], data["Zone 1 Power Consumption"], 'g-')
plt.plot(data["DateTime"], data["Zone 2  Power Consumption"], 'r-')
plt.plot(data["DateTime"], data["Zone 3  Power Consumption"], 'b-')
# Zone 1 is coloured green, Zone 2 red, Zone 3 blue
plt.show()
