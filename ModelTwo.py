import pandas as pd
from  matplotlib import *
from sklearn.ensemble import RandomForestClassifier
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

#Using randomizedSearchCV to tune hyperparameters

param_dist = {"n_estimators": randint(50,500),
              "max_depth": randint(1,20)}


rf = RandomForestClassifier(random_state=42)

rand_search = RandomizedSearchCV(rf, 
                                param_distributions=param_dist,
                                n_iter=5,
                                cv=5, random_state=42)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print(rand_search.best_params_)
#Best parameters are Max_depth 19, N_estimators 286
