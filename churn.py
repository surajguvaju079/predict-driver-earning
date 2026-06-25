from sklearn.tree import (DecisionTreeClassifier,export_text)
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import (confusion_matrix,accuracy_score,classification_report)
model = DecisionTreeClassifier()
df  = pd.read_csv("driver_churn.csv");
X=df[["loads_completed","last_login_days","rating"]]
y=df["churn"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
model.fit(X_train,y_train);
test_prediction = model.predict(X_test);
print(y_test.tolist())
print(test_prediction.tolist())
score = confusion_matrix(y_test,test_prediction)

prediction = model.predict(pd.DataFrame({
    "loads_completed":[6],
    "last_login_days":[25],
    "rating":3.6
}))
print(prediction)
print(score)

print("Accuracy")
print(accuracy_score(y_test,test_prediction));

print("\nClassification Report")
print(classification_report(y_test,test_prediction))

print(export_text(model,feature_names=list(X.columns)))
