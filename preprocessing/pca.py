import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Tetuan City power consumption.csv')
df = df.drop(['diffuse flows'], axis=1)
df = df.drop(['general diffuse flows'], axis=1)
# df = df.drop(['Humidity'], axis=1)
df = df.drop(['Temperature'], axis=1)


x = df.iloc[:, 1:3]
print("x:")
print(x)
print()
x = StandardScaler().fit_transform(x)

pca = PCA(n_components="mle")
x_new = pca.fit_transform(x)

print(x_new)
print(len(x_new[1]))
print(sum(pca.explained_variance_ratio_))
