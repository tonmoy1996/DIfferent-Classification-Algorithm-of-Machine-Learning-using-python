
#Logistic Regression in Python

import pandas as pd

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']

data=pd.read_csv('C:\\Users\\sahat\\Desktop\\Machine Learning\\Regression\\dataset\\diabetes.csv',names=col_names,header= None)
print(data.head())

feature_name=['pregnant', 'insulin', 'bmi', 'age','glucose','bp','pedigree']
print (feature_name)

new_data= data.loc[1:, :]

X=new_data[feature_name]
y=new_data.label
print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train,y_test= train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.linear_model import LogisticRegression

logis=LogisticRegression()

# fit the model with data

logis.fit(X_train,y_train)

#  predict 


y_pred=logis.predict(X_test)


print(y_pred)


#Model Evaluation using Confusion Matrix


from sklearn import metrics

cnf_metrix= metrics.confusion_matrix(y_test,y_pred)

print(cnf_metrix)

print(metrics.accuracy_score(y_test,y_pred))



import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 


#Visualizing Confusion Matrix using Heatmap
 
#class_names=[0,1] # name  of classes
#fig, ax = plt.subplots()
#tick_marks = np.arange(len(class_names))
#plt.xticks(tick_marks, class_names)
#plt.yticks(tick_marks, class_names)
## create heatmap
#sns.heatmap(pd.DataFrame(cnf_metrix), annot=True, cmap="YlGnBu" ,fmt='g')
#ax.xaxis.set_label_position("top")
#plt.tight_layout()
#plt.title('Confusion matrix', y=1.1)
#plt.ylabel('Actual label')
#plt.xlabel('Predicted label')
#plt.savefig(fig)

#Confusion Matrix Evaluation Metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred,pos_label=y_pred[0]))
print("Recall:",metrics.recall_score(y_test, y_pred))

metrics.recall_score(y_test, y_pred,pos_label=y_pred[0])














