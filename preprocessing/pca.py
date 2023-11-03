import pandas as pd
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

"""
df = pd.read_csv('Tetuan City power consumption.csv')
df = df.drop(['diffuse flows'], axis=1)
df = df.drop(['general diffuse flows'], axis=1)
"""


pca = PCA(n_components="mle")


def pca_components(df):
    # df = StandardScaler().fit_transform(df)

    df = pca.fit_transform(df)
    print("PCA explained variance:")
    print(pca.explained_variance_)
    return pd.DataFrame(data=df, columns=pca.get_feature_names_out())


"""
x = df.iloc[:, 1:4]
print("x:")
print(x)
print()
x = StandardScaler().fit_transform(x)

pca = PCA(n_components="mle")
x_new = pca.fit_transform(x)

print(x_new)
print(len(x_new[1]))
print(sum(pca.explained_variance_ratio_))
"""
