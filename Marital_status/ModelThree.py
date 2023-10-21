import pandas as pd
from  matplotlib import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint


df = pd.read_csv("HR_Analytics.csv")

df.isna().sum().sort_values()
df = df.dropna(subset="YearsWithCurrManager")


y = df["MaritalStatus"]
df = df.drop("MaritalStatus", axis=1)

Dummie_values = pd.get_dummies(df, drop_first=True, dtype=int)
X = Dummie_values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Testing if KNN model is better suited
knn = KNeighborsClassifier()

param_dist = {"n_neighbors": randint(1,100),
              "weights": ["uniform", "distant"]}

rand_search = RandomizedSearchCV(knn, 
                                param_distributions=param_dist,
                                n_iter=5,
                                cv=5, random_state=42)


 
rand_search.fit(X_train, y_train)

y_pred = rand_search.predict(X_test)
print(rand_search.best_params_)
#Best params are n_neighbors 72, weights: uniform
print(rand_search.best_score_)
#Best param is only 45%

#Accuracy of only 0.31 without tuning
#print(accuracy_score(y_pred, y_test))