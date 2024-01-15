import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
# Loading the dataset
data = pd.read_csv("Autism-Adult-Data.csv", na_values=["?"])
data_rv = data.drop("age_desc", axis=1)
print("the data attributes are: ")
print(data_rv.columns)
#filling empty data
data_rv["ethnicity"].fillna("missing", inplace=True)
data_rv["relation"].fillna("missing", inplace=True)
#Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data_encode = data_rv.apply(le.fit_transform)
#print("##########After Label Encoding###########")
#print(data_encode)
X = data_encode.iloc[:, :19].values
y = data_encode.iloc[:, 19].values
#print("#########Features#######################")
#print(X)
#print("####################lebel#######################")
#print(y)
#print("###########################################")
#print("data_encode.head()/n", data_encode.head())
print("data_encode.info()/n",data_encode.info())
print("data_encode.describe()/n",data_encode.describe())
# # For plotting histogram
data_encode.hist(bins=50, figsize=(20, 15))
plt.show()
#corr_matrix1 = data_encode.corrwith("ASD")
print(data_encode[data_encode.columns[1:]].corr()['ASD'].sort_values(ascending=False))
#corr_matrix['austim'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
#attributes = X.columns
#scatter_matrix(data_encode)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X, y, test_size = .2, random_state = 42)
from sklearn.linear_model import LogisticRegression
logcls = LogisticRegression()
logcls.fit(X_train,y_train)
y_prd = logcls.predict(X_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(list(y_test), list(y_prd))
rmse = np.sqrt(mse)
print("Logistic Regression RMSE = ", rmse)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
y_prd = clf.predict(X_test)
mse = mean_squared_error(list(y_test), list(y_prd))
rmse = np.sqrt(mse)
print(" Decision Tree RMSE= ", rmse)
from sklearn.svm import SVC
clfsvc = SVC(gamma='auto')
clfsvc.fit(X_train,y_train)
y_prd1 = clfsvc.predict(X_test)
svcmse = mean_squared_error(list(y_test), list(y_prd))
svcrmse = np.sqrt(mse)
print("SVM RMSE = ", svcrmse)