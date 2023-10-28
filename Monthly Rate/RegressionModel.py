import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso
from sklearn.preprocessing import StandardScaler 

df = pd.read_csv("insurance.csv")

#Checks for missing values
print(df.isna().sum().sort_values())

df = pd.get_dummies(df, drop_first=True, dtype=int)

#Finds relationship between categories
plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), annot = True)
#plt.show()
#Finds smokers have highest relationship, then age then BMI will focus on those

y = df["charges"].values
X = df.drop("charges", axis=1).values



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

#Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Ran LinearRegression scoring 0.7344662
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred  = lr.predict(X_test)
test_score_lr = lr.score(X_test, y_test)
print("The test score for lr model is {}".format(test_score_lr))

#Then Ridge scoring 0.7356232
ridgeReg = Ridge(alpha=10)
ridgeReg.fit(X_train,y_train)
test_score_ridge = ridgeReg.score(X_test, y_test)
print("The test score for ridge model is {}".format(test_score_ridge))