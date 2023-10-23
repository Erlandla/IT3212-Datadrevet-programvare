from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from operator import itemgetter


def rfe(X, y):
    estimator = SVR(kernel="linear")
    selector = RFE(estimator, n_features_to_select=4, step=1)
    selector = selector.fit(X, y)
    features = X.columns.to_list()
    for x, y in (sorted(zip(selector.ranking_, features), key=itemgetter(0))):
        print(x, y)
    return selector.transform(X)
