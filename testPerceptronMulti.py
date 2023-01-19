import PerceptronMulti as pm
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[0:149, :4].values,
    df.iloc[0:149, 4].values,
    test_size=.25,
    random_state=1
)

ppn = pm.PerceptronMulti(eta=0.1, n_iter=10)
ppn.fit(X_train, y_train)

score = 0
for i in range(0, len(y_test)):
    if y_test[i] in ppn.predict(X_test[i]):
        score += 1

print(score/len(y_test))
