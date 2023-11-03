import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

data = pd.read_csv("Tetuan City power consumption.csv")

corr_matrix = data.corr()

plt.figure(figsize=(9, 9))
sns.heatmap(corr_matrix, annot=True, cmap="inferno")
plt.show()
