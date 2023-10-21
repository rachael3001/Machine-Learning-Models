import pandas as pd
import numpy as np
from  matplotlib import *
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.ensemble import GradientBoostingClassifier
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

gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

parameters = {
    "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    "min_samples_split": np.linspace(0.1, 0.5, 12),
    "min_samples_leaf": np.linspace(0.1, 0.5, 12),
    "max_depth":[3,5,8],
    "n_estimators":[10]
    }

rand_search = RandomizedSearchCV(gbc, 
                                param_distributions=parameters,
                                n_iter=5,
                                cv=5, random_state=42)

rand_search.fit(X_train, y_train)

best_rf = rand_search.best_estimator_
print(rand_search.best_params_, rand_search.best_score_)

#Best params are Max_depth 19 and n_estimators 264


