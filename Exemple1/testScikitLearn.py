import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :4].values,
    df.iloc[:, 4].values,
    test_size=.25,
    random_state=0
)

k_neighbors = KNeighborsClassifier(n_neighbors=3)
k_neighbors.fit(X_train, y_train)
print("K Neighbors : \t" + str(k_neighbors.score(X_test, y_test)))

ppn = Perceptron()
ppn.fit(X_train, y_train)
print("Perceptron : \t" + str(ppn.score(X_test, y_test)))

svc = SVC(kernel='poly')
svc.fit(X_train, y_train)
print("SVC : \t\t\t" + str(svc.score(X_test, y_test)))
