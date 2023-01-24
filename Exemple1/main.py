import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.linear_model import Perceptron
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('C:\src\Work\DL_2023\Exemple1\Video_Games_Sales.csv')

# On transforme les ventes pour en faire un problème de classification
transformer = Binarizer(threshold=1.0)
np_sales = transformer.transform(df['Global_Sales'].to_numpy().reshape(-1, 1))
df['Global_Sales'] = pd.DataFrame(np_sales)

# Beaucoup de données null, trop pour les remplacer !
df = df.dropna(axis=0)

df.loc[df.loc[:, "Rating"] == "AO", "Rating"] = "M"
df.loc[df.loc[:, "Rating"] == "K-A", "Rating"] = "E"
df.loc[df.loc[:, "Rating"] == "RP", "Rating"] = "T"
df.loc[df.loc[:, "Rating"] == "E10+", "Rating"] = "E"

target = df[['Global_Sales']].values
features = df[['Platform', 'Genre', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count', 'Rating']]

preprocess = make_column_transformer(
    (StandardScaler(), ['Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']),  # on standardise celles-ci
    (OneHotEncoder(), ['Platform', 'Genre', 'Rating'])  # celles-ci en one-hot
)

X_train, X_test, y_train, y_test = train_test_split(features, target, random_state=0)
model = make_pipeline(preprocess, Perceptron(max_iter=400, eta0=0.1, random_state=0))
model.fit(X_train, y_train.ravel())
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
