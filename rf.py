import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

df=pd.read_csv("heart_cleveland_upload.csv")
# print(df)
# print(df["condition"].value_counts())

X=df.drop('condition',axis=1)
y=df['condition']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)

model=RandomForestClassifier()
model.fit(X_train,y_train)

predictions=model.predict(X_test)
print(predictions)

print(model.score(X_test,y_test))

cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(6, 5))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# print(model.n_estimators)