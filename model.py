# -*- coding: utf-8 -*-
"""
Created on Sat May 21 20:03:32 2022

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
import seaborn as sns
colors=sns.color_palette()
from numpy import percentile
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")


#loading the  data
data=pd.read_excel(r"C:\Users\hp\Downloads\final dataset created.xlsx")
data.duplicated().sum()#no duplicate value

##drop irrelevant column
data = data.iloc[:,[3,4,5,7,9,10,11,13,14,15,16,17,18,20]]


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

#Graphical visualisation for categorical column
 #balance ratio male and female
sns.countplot(data['Name_Of_Test'])
sns.countplot(data['Sample'])
sns.countplot(data['Way_Of_Storage_Of_Sample'])
sns.countplot(data['Cut-off Schedule'])
sns.countplot(data['Traffic_Conditions'])
sns.countplot(data['Reached_On_Time']) #Advance storage not reach on time

#Value count
data['Name_Of_Test'].value_counts()
data['Way_Of_Storage_Of_Sample'].value_counts() #Normal -967 And Advanced-52
data['Reached_On_Time'].value_counts() #Y-956 And N-63 # most of advance storage not reach on time

# outliers detection
sns.boxplot(data=data.iloc[:,[4,6,8]]) #outlier present 
sns.boxplot(data=data.iloc[:,[9,10,11,12]]) #outlier present

#remove outlier by percentile method
columns = data.columns
for j in columns:
    if isinstance(data[j][0], str) :
        continue
    else:
        #defining quartiles
        quartiles = percentile(data[j], [25,75])
        # calculate min/max
        lower_fence = quartiles[0] - (1.5*(quartiles[1]-quartiles[0]))
        upper_fence = quartiles[1] + (1.5*(quartiles[1]-quartiles[0]))
        data[j] = data[j].apply(lambda x: upper_fence if x > upper_fence else (lower_fence if x < lower_fence else x))

#for advance visualization
sns.pairplot(data) 


#Preprocessing

#Create dummies for categorical column
data1 = pd.get_dummies(data.iloc[:,: ], drop_first = True)

#Normalization 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)

data1= norm_func(data1)

#Train and test split

X = data1.iloc[:,:24] #Independent variable 
y = data1.iloc[:,-1] #Dependent variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state = 21,stratify=y)
 #our data set is highly imbalance so that we use stratify sample
 
print(X_train.shape)
print(X_test.shape)
#Checking training and testing sample  equal or not
print(y_train.value_counts(normalize = True).round(2))
print(y_test.value_counts(normalize = True).round(2)) #train and test sample are equal Y- 0.94  N-0.06



#Using SMOTE for  imbalance dataset

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42, sampling_strategy = 0.75)
X_res, y_res = sm.fit_resample(X_train, y_train)


#Model building

#Model1 -Logistic regression 
#We use logistic regression because our targest variable in binary class
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',
                      max_iter=5000,
                      solver='liblinear',
                      random_state=25) 
 
lr.fit(X_res,y_res) #Fit and train the model
y_pred=lr.predict(X_test) 

confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)  #0.9901960784313726 
print(classification_report(y_test, y_pred))

#Calculating test and train accuracy for model 1
train_pred =lr.predict(X_train)
accuracy_score(train_pred,y_train) #0.9889570552147239


#Saving the model to disk
import pickle
pickle.dump(lr,open('model.pkl','wb'))

#Loading model to compare the result
model = pickle.load(open('model.pkl','rb'))
