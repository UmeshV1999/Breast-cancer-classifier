#Made By Umesh Verma
#uploaded at https://uskai.com/2019/05/01/building-a-breast-cancer-classifier-using-machine-learning-in-5-minutes/
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline 

#this is magic function which plots graph directly below the code cell that produced it
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
#Import Cancer data from Sklearn library
#OR
#Dataset can is also available at(https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
cancer
df_cancer=pd.DataFrame(np.c_[cancer['data'],cancer['target']],columns=np.append(cancer['feature_names'],['target']))
df_cancer.head()
#Let's plot out just the first 5 variables (features)
sns.pairplot(df_cancer,hue='target',vars=['mean radius', 'mean texture','mean perimeter','mean area','mean smoothness'])
# Checking the co-relation between features
plt.figure(figsize=(20,12))
sns.heatmap(df_cancer.corr(), annot=True)
X=df_cancer.drop(['target'],axis=1) #we are dropping target column from dataframe
X.head()
Y=df_cancer['target']
Y.head()
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#train test split
X_train, X_test, Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=42)
X_train_min=X_train.min()
X_train_max=X_train.max()
#Normalizing
X_train_range=(X_train_max - X_train_min)
X_train_scaled=(X_train - X_train_min)/(X_train_range)
X_train_scaled.head()
X_test_min=X_test.min()
X_test_max=X_test.max()
X_test_range = (X_test_max-X_test_min)
X_test_scaled=(X_test-X_test_min)/X_test_range
# importing SVM model
from sklearn.svm import SVC
svc_model=SVC()
svc_model.fit(X_train_scaled,Y_train)
Y_predict=svc_model.predict(X_test_scaled)
# confusion matrix for our classifier's performance
from sklearn.metrics import classification_report, confusion_matrix
cm=confusion_matrix(Y_test,Y_predict)
cm=np.array(confusion_matrix(Y_test,Y_predict, labels=[1,0]))
confusion=pd.DataFrame(cm,index=['is_cancer','is_healthy'],columns=['predicted_cancer','predicted_healthy'])
confusion
print(classification_report(Y_test,Y_predict))