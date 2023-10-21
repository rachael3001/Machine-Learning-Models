import pandas as pd
from  matplotlib import *
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


df = pd.read_csv("HR_Analytics.csv")

#Check for missing values
df.isna().sum().sort_values()
df = df.dropna(subset="YearsWithCurrManager")

y = df["JobRole"]
df = df.drop("JobRole", axis=1)

#Convert Categorical data to usable and to get our data
Dummie_values = pd.get_dummies(df, drop_first=True, dtype=int)
X = Dummie_values

#Split data using a 10% split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

#Using RandomForest, hyper-parameters tuned from modelTwo
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

param_dist = {"n_estimators": randint(50,500),
              "max_depth": randint(1,20)}

rand_search = RandomizedSearchCV(rf, 
                                param_distributions=param_dist,
                                n_iter=5,
                                cv=5, random_state=42)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print(rand_search.best_params_, rand_search.best_score_)

#Best params are Max_depth 19 and n_estimators 264


