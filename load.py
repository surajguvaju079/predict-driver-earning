import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

model = LinearRegression()
df = pd.read_csv("loads.csv");
X=df[["loads_completed"]]
y=df["earning"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42);

model.fit(X_train,y_train)
prediction = model.predict(X_test);
score = r2_score(y_test,prediction)
print(model.predict([[12]]))
print(model.predict([[14]]))
print(score)
print(model.coef_)
print(model.intercept_)





