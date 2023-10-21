import pandas as pd
from  matplotlib import *
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn._config import get_config, set_config

#Tried Gradient boosting model accuracy increased to 

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

rf = GradientBoostingClassifier(random_state=42)
rf.fit(X_train, y_train)

set_config(print_changed_only=False)
y_pred = rf.predict(X_test)
print(rf)

#Score of 59% before tuning and 65% aftr tuning
print(accuracy_score(y_test, y_pred))

