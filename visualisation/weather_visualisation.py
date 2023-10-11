import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_csv("Tetuan City power consumption.csv")

dateTime = data["DateTime"]

temp = data["Temperature"]
windSpeed = data["Wind Speed"]
diffuseFlows = data["diffuse flows"]
generalDiffuseFlows = data["general diffuse flows"]

plt.xticks(np.arange(0, len(data["DateTime"])-1, (len(data["DateTime"])-1)/12))

figure, axis = plt.subplots(4,1)

# --- TEMPERATURE ---
axis[0].set_title("Temperature over time")
axis[0].set_xticks(np.arange(0, len(dateTime)-1, (len(dateTime)-1)/12))
axis[0].plot(dateTime, temp)

# --- WIND SPEED ---
axis[1].set_title("Wind speed over time")
axis[1].set_xticks(np.arange(0, len(dateTime)-1, (len(dateTime)-1)/12))
axis[1].plot(dateTime, windSpeed)

# --- DIFFUSE FLOWS ---
axis[2].set_title("Diffuse flows over time")
axis[2].set_xticks(np.arange(0, len(dateTime)-1, (len(dateTime)-1)/12))
axis[2].plot(dateTime, diffuseFlows)

# --- GENERAL DIFFUSE FLOWS ---
axis[3].set_title("General diffuse flows over time")
axis[3].set_xticks(np.arange(0, len(dateTime)-1, (len(dateTime)-1)/12))
axis[3].plot(dateTime, generalDiffuseFlows)

plt.subplots_adjust(hspace=0.5)
plt.show()