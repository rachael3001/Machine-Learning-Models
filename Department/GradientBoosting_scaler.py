import pandas as pd
from  matplotlib import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn._config import set_config
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

my_pipeline = Pipeline(steps=[("Preprocesser", StandardScaler()), ("model", GradientBoostingClassifier())])
scores = cross_val_score(my_pipeline, X, y, cv=10, scoring ='accuracy')
print(scores)

#Scores range from 0.67 to 0.60




#X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1, random_state=42)

#gbc = GradientBoostingClassifier(random_state=42)
#gbc.fit(X_train, y_train)

#set_config(print_changed_only=False)
#y_pred = gbc.predict(X_test)

#Score of 59% before tuning and 65% aftr tuning
#print(accuracy_score(y_test, y_pred))



