# -*- coding: utf-8 -*-
"""
Created on Thu May 26 16:21:18 2022

@author: hp
"""

#OBJECTIVE :-To ensure integrity and stability of biological home samples during the transport process.
#Minimise - Sample loss
#Maximise- The quality service delivered to patients while respecting several constraints
#Constrain- Scheduling and Routing of logistic


#import dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


#loading the  data
data=pd.read_excel(r"C:\Users\hp\Desktop\Project Aditi 69\sampla_data_08_05_2022(final).xlsx")
data.duplicated().sum()#no duplicate value

##drop irrelevant column
data.drop(["Patient_ID","Test_Booking_Date","Sample_Collection_Date", "Mode_Of_Transport"], axis = 1, inplace = True)

#Know about data
data.describe()

#checking null values
data.isna().sum() #no null values
data.info()



#EDA
#Measure of central tendency
# 1.Mean
data.mean() #avg cutt off time =13 ,time taken for sample=4 and lab loction is 54
#2.Median
data.median() #avg cutt of time=17 , and lab location=10 #outlier is present

#Measure of dispersion
# 1.variance
data.var()

# 2.Standard deviation
data.std()

# 3.Skewness
data.skew()

# 4.Kurtosis
data.kurt()

# Visualisation for numerical column
data.hist()


#Preprocessing

#Convert categorical to numeric column

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()
data["Patient_Gender"] = label.fit_transform(data["Patient_Gender"])
data["Test_Name"] = label.fit_transform(data["Test_Name"])
data["Sample"] = label.fit_transform(data["Sample"])
data["Way_Of_Storage_Of_Sample"] = label.fit_transform(data["Way_Of_Storage_Of_Sample"])
data["Cut-off Schedule"] = label.fit_transform(data["Cut-off Schedule"])
data["Traffic_Conditions"] = label.fit_transform(data["Traffic_Conditions"])



#Train and test split

X = data.iloc[:,:-1] #Independent variable 
y = data.iloc[:,-1] #Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 42,stratify=y)
 #our data set is imbalance so that we use stratify sample
 
print(X_train.shape)
print(X_test.shape)
#Checking training and testing sample  equal or not
print(y_train.value_counts(normalize = True).round(2))
print(y_test.value_counts(normalize = True).round(2)) #train and test sample are equal Y- 0.94  N-0.06



#Model building

#Model1 -Logistic regression 
#We use logistic regression because our targest variable in binary class
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',
                      max_iter=2500,
                      solver='liblinear',
                      random_state=0) 
 
lr.fit(X_train,y_train) #Fit and train the model
y_pred=lr.predict(X_test) 

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)  #0.0.9803
print(classification_report(y_test, y_pred))

#Calculating test and train accuracy for model 1
train_pred =lr.predict(X_train)
accuracy_score(train_pred,y_train) #0.0.0.9779



# saving the model

import pickle

pickle.dump(lr, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))


