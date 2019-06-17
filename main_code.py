# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 03:27:21 2019

@author: kemal
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set() #set plot style
import missingno as msno #for displaying missing values 
import scipy.stats as stats #for plotting normality
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE




# Importing data
rain_data = pd.read_csv("C:\\Users\\kemal\\OneDrive\\Masaüstü\\Data Analytics MSc\\Spec 9270 - Machine Learning\\Assignments\\Task 1\\Rain Prediction\\weatherAUS.csv", engine="python")
# Exclude RISK_MM because it leakes information to the model
rain_data = rain_data.drop(columns="RISK_MM") 

#####DATA UNDERSTANDING#####

# Descriptive statistics
rain_data.describe() 
rain_data.info() #summary 

# Yes,No counts for the target (to see if the target variable is imbalanced or not)
sns.countplot(rain_data["RainTomorrow"]) # target variable is infact imbalanced, it is going to be taken care of in the preperation section
yes=np.count_nonzero(rain_data["RainTomorrow"]=="Yes")*100/len(rain_data)
no=np.count_nonzero(rain_data["RainTomorrow"]=="No")*100/len(rain_data)

# Noramlity check
numeric_cols=rain_data.drop(columns=["Date","Location","WindGustDir","WindDir9am","WindDir3pm","RainToday","RainTomorrow"])
numeric_cols.hist(bins=50, figsize=(25,20))
cols = numeric_cols.columns
k = 0
plt.figure("normality")
for i in range(2):
    for j in range(7):
        ax = plt.subplot2grid((2,7), (i,j))
        stats.probplot(numeric_cols[cols[k]], plot=plt)
        ax.set_title(cols[k])
        k = k+1
plt.subplots_adjust(hspace=0.95)
plt.show()

# Visualisation for categorical attributes
sns.countplot(rain_data["WindGustDir"])
sns.countplot(rain_data["WindDir9am"])
sns.countplot(rain_data["WindDir3pm"])
sns.countplot(rain_data["RainToday"])

# Correlation Matrix (to see whether multicollinearity exists in the dataset)
corr_rain = rain_data.corr() # Some variables are highly correlated with eachother, it is going to be taken care of in the preperation section
sns.heatmap(corr_rain,annot=True,linewidths=0.25)

# Exploring missing values
null_counts = rain_data.isnull().sum()
perc_null = 100*null_counts/len(rain_data)
msno.bar(rain_data)

#####DATA PREPERATION#####

## Split dataset - Dividing the dataset into training and test dataset

X=rain_data.iloc[:,0:22] # predictors
y=rain_data.iloc[:,22:23] # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35,random_state=5)

## Dimensionality reduction - removing correlated attributes

#Correlation matrix
corr_rain = X_train.corr() 
sns.heatmap(corr_rain,annot=True,linewidths=0.25)

# Calculating average correlation for highly correlated attributes
avg_temp3pm= sum(corr_rain["Temp3pm"])/len(corr_rain)
avg_maxtemp = sum(corr_rain["MaxTemp"])/len(corr_rain) # MaxTemp has higher average correlation
avg_pressure9am = sum(corr_rain["Pressure9am"])/len(corr_rain)
avg_pressure3pm = sum(corr_rain["Pressure3pm"])/len(corr_rain) # Pressure3pm has higher average correlation
avg_temp9am= sum(corr_rain["Temp9am"])/len(corr_rain)
avg_mintemp= sum(corr_rain["MinTemp"])/len(corr_rain) # MinTemp has higher average correlation
avg_temp9am= sum(corr_rain["Temp9am"])/len(corr_rain)
avg_temp3pm= sum(corr_rain["Temp3pm"])/len(corr_rain) # Temp9am has higher average correlation
# Remove one of the two highly correlated atrribute which has highest average correlation with other attribute
X_train=X_train.drop(columns=["MaxTemp","Pressure3pm","MinTemp","Temp9am"])
X_test=X_test.drop(columns=["MaxTemp","Pressure3pm","MinTemp","Temp9am"])
corr_rain = X_train.corr()
sns.heatmap(corr_rain,annot=True,linewidths=0.25)

## Handling missing values

# Concatenate predictors with target variable - Send them to R for handling missing values
rain_to_R_train=pd.concat([X_train,y_train],axis=1,sort=False)
rain_to_R_test=pd.concat([X_test,y_test],axis=1,sort=False)

rain_to_R_train.to_csv("C:\\Users\\kemal\\OneDrive\\Masaüstü\\Data Analytics MSc\\Spec 9270 - Machine Learning\\Assignments\\Task 1\\Rain Prediction\\rain_to_R_train.csv")
rain_to_R_test.to_csv("C:\\Users\\kemal\\OneDrive\\Masaüstü\\Data Analytics MSc\\Spec 9270 - Machine Learning\\Assignments\\Task 1\\Rain Prediction\\rain_to_R_test.csv")

## Read train and test data from R after imputations

train_R = pd.read_csv("C:\\Users\\kemal\\OneDrive\\Masaüstü\\Data Analytics MSc\\Spec 9270 - Machine Learning\\Assignments\\Task 1\\Rain Prediction\\Rain_Prediction_R\\rain_from_R_train.csv", engine="python")
test_R = pd.read_csv("C:\\Users\\kemal\\OneDrive\\Masaüstü\\Data Analytics MSc\\Spec 9270 - Machine Learning\\Assignments\\Task 1\\Rain Prediction\\Rain_Prediction_R\\rain_from_R_test.csv", engine="python")

train_R=train_R.iloc[:,1:20] # remove unnanmed column
test_R=test_R.iloc[:,1:20] # remove unnanmed column

train_R = train_R.set_index("Date") # assign date as index in training dataset
test_R = test_R.set_index("Date") # assign date as index in testing dataset

## Binarization

# Change RainToday and RainTomorrow from Yes/No to 1 and 0

train_R["RainToday"].replace({"No":0,"Yes":1},inplace=True)
train_R["RainTomorrow"].replace({"No":0,"Yes":1},inplace=True)
test_R["RainToday"].replace({"No":0,"Yes":1},inplace=True)
test_R["RainTomorrow"].replace({"No":0,"Yes":1},inplace=True)

## Scaling

#Seperate predictors from target
X_train=train_R[train_R.columns.difference(["RainTomorrow"])]
y_train=train_R.iloc[:,17:18]

X_test=test_R[test_R.columns.difference(["RainTomorrow"])]
y_test=test_R.iloc[:,17:18]

# Seperate the categorical variables

categorical_train=X_train[["Location","WindGustDir","WindDir9am","WindDir3pm"]]
X_train=X_train.drop(columns=["Location","WindGustDir","WindDir9am","WindDir3pm"])
col_names=X_train.columns

categorical_test=X_test[["Location","WindGustDir","WindDir9am","WindDir3pm"]]
X_test=X_test.drop(columns=["Location","WindGustDir","WindDir9am","WindDir3pm"])

#Label Encoding for Location
label_enc=LabelEncoder()
categorical_train["Location"]=label_enc.fit_transform(categorical_train["Location"])
categorical_test["Location"]=label_enc.transform(categorical_test["Location"])

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


#Change categorical variable to dummy variables

X_train_dummy=pd.get_dummies(categorical_train)
X_test_dummy=pd.get_dummies(categorical_test)

# Bind categorical and numerical attributes together

X_train=pd.DataFrame(X_train, columns = col_names)
X_train=X_train.set_index(train_R.index)
X_test=pd.DataFrame(X_test, columns = col_names)
X_test=X_test.set_index(test_R.index)

X_train=pd.concat([X_train,X_train_dummy],axis=1)
X_test=pd.concat([X_test,X_test_dummy],axis=1)
col_names_all=X_train.columns


## Oversampling

sm = SMOTE(random_state=12, ratio = 1.0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
# Yes,No counts for the target (to see if the target variable is imbalanced or not)
#sns.countplot(y_train_res) # target variable is infact imbalanced, it is going to be taken care of in the preperation section

# For convention the names changed back to their original
X_train,y_train=X_train_res,y_train_res


#####MODELING#####

## Random forest

# Train Random Forest model and tune it
random_forest=RandomForestClassifier(random_state=5)
n_estimators_forest=[50,100,200]
max_depth=range(10,40)
hyperparameters=dict(n_estimators=n_estimators_forest,max_depth=max_depth)
random_search_forest=GridSearchCV(random_forest,hyperparameters,scoring="balanced_accuracy",cv=5)
random_search_forest.fit(X_train,y_train)
best_rand_model=random_search_forest.best_estimator_ #save best model

# Test Random forest model
pred_test_forest=random_search_forest.predict(X_test)
conf_forest=metrics.confusion_matrix(y_test, pred_test_forest)
metrics.balanced_accuracy_score(y_test, pred_test_forest)

# Train the model again with best estimators and subset of features
forest_fea_imp=random_search_forest.best_estimator_.feature_importances_.T
forest_fea_imp=forest_fea_imp[:,np.newaxis]
forest_fea_imp=pd.DataFrame(forest_fea_imp.T,columns=col_names_all) #convert numpy to pandas to attach the column names to see which attribute is the most important
forest_fea_imp=forest_fea_imp.sort_values(ascending=False,by=[0],axis=1) #scale features according to their importance

# Try the Random forest model and fine tune feature set from 1 to all(scaled according to their importance) and select the best feature set
train_bal_acc = [] 
test_bal_acc = []
bal_acc= 0
opt_features=0
for i in range(1,63):
    
    col_names_forest_imp=forest_fea_imp.columns[0:i] #Columns that will be used to train the model
    best_random_forest=RandomForestClassifier(max_depth=15,n_estimators=200,random_state=5)
    X_train=pd.DataFrame(X_train,columns=col_names_all) #convert X_train back to pandas and add columns to select the best columns
    best_random_forest.fit(X_train[col_names_forest_imp],y_train)
    print(i,".training finished")
    pred_train_best_forest=best_random_forest.predict(X_train[col_names_forest_imp])
    train_bal_acc.append(metrics.balanced_accuracy_score(y_train, pred_train_best_forest)) #add the accuracy of ith feature set to train array
    
    # Test Random forest model
    X_test=pd.DataFrame(X_test,columns=col_names_all) #convert X_train back to pandas and add columns to select the best columns
    pred_test_best_forest=best_random_forest.predict(X_test[col_names_forest_imp])
    test_bal_acc.append(metrics.balanced_accuracy_score(y_test, pred_test_best_forest)) #add the accuracy of ith depth to test array
    
    if bal_acc<metrics.balanced_accuracy_score(y_test, pred_test_best_forest):# Select most accurate(balanced accuracy) feature set
        bal_acc=metrics.balanced_accuracy_score(y_test, pred_test_best_forest)
        opt_features = i # To hold the optimum features

# Plotting the balanced accuracy and iterations with different depths
iterations = np.array(range(i)) 
plt.plot(iterations, train_bal_acc)
plt.plot(iterations, test_bal_acc)
plt.xlabel("Features of Random Forest")
plt.ylabel("Balanced Accuracy")
plt.legend(["Training Bal_Accuracy","Test Bal_Accuracy"])

# Best model with optimum feature set
col_names_forest_imp=forest_fea_imp.columns[0:opt_features] #Columns that will be used to train the model
best_random_forest=RandomForestClassifier(max_depth=15,n_estimators=200,random_state=5)
X_train=pd.DataFrame(X_train,columns=col_names_all) #convert X_train back to pandas and add columns to select the best columns
best_random_forest.fit(X_train[col_names_forest_imp],y_train)
pred_train_best_forest=best_random_forest.predict(X_train[col_names_forest_imp]) 
pred_test_best_forest=best_random_forest.predict(X_test[col_names_forest_imp]) 
conf_forest=metrics.confusion_matrix(y_test, pred_test_best_forest)
metrics.balanced_accuracy_score(y_train, pred_train_best_forest)
metrics.balanced_accuracy_score(y_test, pred_test_best_forest)
metrics.accuracy_score(y_test, pred_test_best_forest)
np.mean(random_search_forest.cv_results_["std_test_score"])

## AdaBoost

#Ada boost with gridsearch (Adaboost-1)
random_search_ada=AdaBoostClassifier(random_state=5)
n_estimators_ada=[100,200,300]
learning_rate=[0.7,1.0,1.5]
hyperparameters=dict(n_estimators=n_estimators_ada,learning_rate=learning_rate)
random_search_ada=GridSearchCV(random_search_ada,hyperparameters,scoring="balanced_accuracy",cv=5)
random_search_ada.fit(X_train,y_train)
best_ada_model=random_search_ada.best_estimator_
pred_train_random_ada=random_search_ada.predict(X_train)
metrics.balanced_accuracy_score(y_train,pred_train_random_ada)
metrics.accuracy_score(y_train,pred_train_random_ada)
pred_test_random_ada=random_search_ada.predict(X_test)
metrics.balanced_accuracy_score(y_test,pred_test_random_ada)
metrics.accuracy_score(y_test,pred_test_random_ada)
conf_random_ada=metrics.confusion_matrix(y_test, pred_test_random_ada)
np.mean(random_search_ada.cv_results_["std_test_score"])


# Adaboost with Random forest model as base model (Adaboost-2) (this random forest model is one
# of the best random forest model(for this data - determined with trial and error) that min the risk of overfitting and max balanced accuracy)
ada=AdaBoostClassifier(RandomForestClassifier(max_depth=10,n_estimators=50,n_jobs=-1),random_state=5)
ada.fit(X_train,y_train)
pred_test_ada=ada.predict(X_test)
conf_ada=metrics.confusion_matrix(y_test, pred_test_ada)
metrics.balanced_accuracy_score(y_test, pred_test_ada)
metrics.accuracy_score(y_test, pred_test_ada)

# Test Adaboost model
pred_test_ada=random_search_ada.predict(X_test)
conf_ada=metrics.confusion_matrix(y_test, pred_test_ada)
metrics.balanced_accuracy_score(y_test, pred_test_ada)
metrics.accuracy_score(y_test, pred_test_ada)

## Logistic Regression

# Logistic Regression model with gridsearch
logistic=LogisticRegression(solver="saga",max_iter=500,random_state=5)
penalty = ['l1', 'l2']
C = [0.5,1.5]
hyperparameters = dict(C=C, penalty=penalty)
grid_Search_log = GridSearchCV(logistic, hyperparameters, cv=5, verbose=0)
grid_Search_log = LogisticRegression(solver="saga",C=1.5,max_iter=500,random_state=5)
grid_Search_log.fit(X_train, y_train)


# Test the logistic regression model
pred_grid_log_test=grid_Search_log.predict(X_test)
metrics.balanced_accuracy_score(y_test,pred_grid_log_test)
np.mean(grid_Search_log.cv_results_["std_test_score"])
    =metrics.confusion_matrix(y_test, pred_grid_log_test)
metrics.balanced_accuracy_score(y_test,pred_grid_log_test)
metrics.accuracy_score(y_test,pred_grid_log_test)
log_best_est=grid_Search_log.best_estimator_


## SVC

#SVC
svc=SVC(kernel="poly",gamma="scale",random_state=5)
svc.fit(X_train,y_train)
svc_train_pred=svc.predict(X_train)
metrics.balanced_accuracy_score(y_train,svc_train_pred)
svc_test_pred=svc.predict(X_test)
metrics.accuracy_score(y_test,svc_test_pred)
metrics.balanced_accuracy_score(y_test,svc_test_pred)
conf_svc=metrics.confusion_matrix(y_test, svc_test_pred)



