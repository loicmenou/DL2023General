import PerceptronMulti as pm
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
print(df.tail())
X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, :4].values,
    df.iloc[:, 4].values,
    test_size=.25,
    random_state=0
)

ppn1 = pm.PerceptronMulti(eta=0.05, n_iter=500)
ppn1.fit_one_vs_one(X_train, y_train)
print("Perceptron one vs one : " + str(ppn1.score_one_vs_one(X_test, y_test)))

ppn2 = pm.PerceptronMulti(eta=0.05, n_iter=500)
ppn2.fit_one_vs_all(X_train, y_train)
print("Perceptron one vs all : " + str(ppn2.score_one_vs_all(X_test, y_test)))
