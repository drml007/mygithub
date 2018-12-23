# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 17:39:20 2018

@author: Gaurav Rai
"""

# Set working directory
#path = input("Input file path directory: ")
import os
#os.chdir(path)
os.chdir('D:\Imarticus\Imarticus_Final_Project')
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')

# Reading complete dataset
dataset = pd.read_csv('XYZCorp_LendingData.txt',sep = '\t',low_memory=False)

# Converting issue_d column to a datetime type and storing in issue_d1
dataset['issue_d1'] = pd.to_datetime(dataset.issue_d)

# Sorting the data based on issue_d1
dataset_sort_issue_d = dataset.sort_values(by=['issue_d1'])

dataset_sort_issue_d.drop(['id','member_id'], axis=1, inplace=True)

# Segregating the data based on Train and Test 
dataset_train = dataset_sort_issue_d.loc[(dataset_sort_issue_d.issue_d1 >= '2007-06-01') & (dataset_sort_issue_d.issue_d1 <= '2015-05-01')].copy()
dataset_test = dataset_sort_issue_d.loc[(dataset_sort_issue_d.issue_d1 >= '2015-06-01')].copy()

# Adding Source column with values Train and Test for easier identification during later stages
dataset_train["Source"] = "Train"
dataset_test["Source"] = "Test"

# Combining the dataset again for further computations
dataset_full = pd.concat([dataset_train,dataset_test], axis = 0)

# Exploring the data
dataset_full.head(5)
dataset_full.tail(5)

# Returns summary for continuous vars
data_desc = dataset_full.describe()

# Check the data types of the columns
data_dtypes = dataset_full.dtypes


dataset_full.drop('issue_d1', axis=1, inplace=True)
dataset_train.drop('issue_d1', axis=1, inplace=True)
dataset_test.drop('issue_d1', axis=1, inplace=True)

########################
# Missing value imputation (NAs)
########################

# First check for NAs
# returns the sum of NA values column wise
b = dataset_full.isnull().sum() 

# For Continuous vars: Impute with median
# For Categorical vars: Impute with mode

# for column verification_status_joint there is no value in train dataset 
# but value is there in test dataset so removing this field from analysis

#dataset_train.to_csv('train_dataset.csv', sep='\t', encoding='utf-8')


for column in dataset_full:
    if(column != 'verification_status_joint'):
        if( (dataset_full[column].dtype == 'object') & (b[column] > 0)):
            dataset_full[column].fillna(dataset_train[column].mode()[0], inplace = True)
        elif ((dataset_full[column].dtype != 'object') & (b[column] > 0)):
            dataset_full[column].fillna(dataset_train[column].median(), inplace = True)

c = dataset_full.isnull().sum() 



#dataset_train.to_csv('full_dataset_train.csv')

a = dataset_train.dtypes


###########################################
## Outlier Detection & Correction - to do 
########################################### 

for column in dataset_train:
    if ((dataset_train[column].dtype != 'object') & ((column != "default_ind") | (column != "issue_d1"))):
        q = dataset_train[column].quantile(0.99)
        r = dataset_train[column].quantile(0.01)
        dataset_full[column] = np.where(dataset_full[column] > q, q, dataset_full[column])
        dataset_full[column] = np.where(dataset_full[column] < r, r, dataset_full[column])
            
            
# In this dataset there are certain columns which contain all 
# nulls (in full dataset or train ) so we need to remove all those columns
for i in c.index:
    if(c[i]>0):
        dataset_full.drop(i, axis=1, inplace=True)
        



#####################################################################
## Creating bins using pd.crosstabs and looking closely at the data
##################################################################### 


for column in dataset_full:
    if(dataset_full[column].dtype == 'object') & (column != 'issue_d'):
        
        print('before >>>>> ',column,'::::',dataset_full[column].nunique())
        a = pd.crosstab(dataset_full[column], dataset_full['default_ind'], normalize='index')

        temp_desc = a.index[a[0] == 0]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0",dataset_full[column])            
            
        
        temp_desc = a.index[a[0] ==1] 
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_1",dataset_full[column])           

        print('after >>>>> ',column,'::::',dataset_full[column].nunique())

# To be tuned further after the first iteration
#emp_title
#desc
#title
#zip_code
#addr_state
#earliest_cr_line
#last_pymnt_d
#last_credit_pull_d

for column in ['emp_title','desc','title','zip_code','addr_state','earliest_cr_line','last_pymnt_d','last_credit_pull_d']:
    if(dataset_full[column].dtype == 'object') & (column != 'issue_d'):
        
        print('before >>>>> ',column,'::::',dataset_full[column].nunique())
        a = pd.crosstab(dataset_full[column], dataset_full['default_ind'], normalize='index')

        temp_desc = a.index[a[0] == 0]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0",dataset_full[column])            
            
        
        temp_desc = a.index[a[0] > 0] & a.index[a[0] <= 0.1]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.1",dataset_full[column]) 
        
        temp_desc = a.index[a[0] > 0.1] & a.index[a[0] <= 0.2]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.2",dataset_full[column])         
        
        temp_desc = a.index[a[0] > 0.2] & a.index[a[0] <= 0.3]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.3",dataset_full[column])  
        
        temp_desc = a.index[a[0] > 0.3] & a.index[a[0] <= 0.4]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.4",dataset_full[column])  

        temp_desc = a.index[a[0] > 0.4] & a.index[a[0] <= 0.5]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.5",dataset_full[column])

        temp_desc = a.index[a[0] > 0.5] & a.index[a[0] <= 0.6]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.6",dataset_full[column])
        
        temp_desc = a.index[a[0] > 0.6] & a.index[a[0] <= 0.7]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.7",dataset_full[column])
        
        temp_desc = a.index[a[0] > 0.7] & a.index[a[0] <= 0.8]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.8",dataset_full[column])
        
        temp_desc = a.index[a[0] > 0.8] & a.index[a[0] <= 0.9]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0.9",dataset_full[column])
        
        temp_desc = a.index[a[0] > 0.9] & a.index[a[0] <= 1]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_1",dataset_full[column])           

        print('after >>>>> ',column,'::::',dataset_full[column].nunique())



# memory management 
del [dataset, dataset_sort_issue_d,dataset_test,dataset_train]

##########################
# Dummy variable creation
##########################       
        
# Create dummy variables from all indep categorical variables

# Step 1: Identify categorical vars
Categ_Vars = dataset_full.loc[:,dataset_full.dtypes == object].columns

# Step 2: Create dummy vars
# Removing desc, title and emp_title to avoid memory issues as these field values have high cardinality 
Dummy_Df = pd.get_dummies(dataset_full[Categ_Vars].drop(['Source'], axis = 1), drop_first=True, dtype=int)
Dummy_Df.columns
Dummy_Df.shape
Dummy_Df.dtypes

# Step 3: Append the Dummy_Df with dataset_full. Call it dataset_full2
dataset_full2 = pd.concat([dataset_full, Dummy_Df], axis = 1)
dataset_full2.shape
dataset_full2.dtypes

# Step 4.1: Drop all the irrelavant and categorical columns (Do NOT drop Source column - We need it for sample splitting)
Cols_To_Drop = Categ_Vars.drop('Source') # Ensure you discard 'Source' column from "columns to drop"

# Step 4.2
dataset_full2.drop(Cols_To_Drop, axis=1, inplace=True)
dataset_full2.shape
dataset_full2.columns

# This step removes all those columns which have only one value in all the records
# they do not add any value to analysis and it creates issues during vif 
# calculation resulting in NaN
aa = []
for column in dataset_full2:
    if(dataset_full2[column].nunique()==1):
        aa.append(column)

dataset_full2.drop(aa, axis=1, inplace=True) 

# This step is to take care of those dummy variables created which have 
# classification split of 0 & 1 of the ratio greater than 99% and 1% and vice versa
# This is to remove dummy variables which will provide better features to do analysis with
aa=[]
for column in dataset_full2:
    if(dataset_full2[column].dtypes == 'int32'):
        bb = dataset_full2.groupby([column]).size()
        if((bb[0]/bb.sum()>0.99) | (1-(bb[0]/bb.sum())<0.01)):
            aa.append(column)

dataset_full2.drop(aa, axis=1, inplace=True)           


# memory management 
del[Dummy_Df, dataset_full]

########################
# Sampling
########################

# intercept = 1 always gets multiplied at the backend and for easier 
# explanability we give value of 1 as its multiplication is easy
dataset_full2['Intercept'] = 1

# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column
Train = dataset_full2.loc[dataset_full2.Source == "Train",:].drop('Source', axis = 1).copy()
Train.shape
Test = dataset_full2.loc[dataset_full2.Source == "Test",:].drop('Source', axis = 1).copy()
Test.shape


# To address those conditions for which train data contains 
# only one category of values and rest of the category has fallen into test data 
for column in Train:
    if(Train[column].nunique()==1 and column!= 'Intercept'):
        Train.drop(column, axis=1, inplace=True)
        Test.drop(column, axis=1, inplace=True)

# memory management 
del[dataset_full2]

########################
# Divide data into Train_X, Train_Y & Test_X, Test_Y
########################

# Divide each dataset into Indep Vars and Dep var
Train_X = Train.drop('default_ind', axis = 1).copy()
Train_Y = Train['default_ind'].copy()
Test_X = Test.drop('default_ind', axis = 1).copy()
Test_Y = Test['default_ind'].copy()

Train_X.shape

########################
# Multicollinearity check
########################

# Check for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor


cols_to_drop_vif = []
# All columns with vif value > 10 were earmarked to be dropped from analysis
for i in range(Train_X.shape[1]-1):
    temp_vif = variance_inflation_factor(Train_X.values, i) # Pass Train_X.values and i (col_number)
    print(Train_X.columns[i], ": ", temp_vif)
    if(temp_vif>10):
        print('Since vif value is greater than 10 so dropping the column ',Train_X.columns[i])
        cols_to_drop_vif.append(Train_X.columns[i])
    
Train_X.drop(cols_to_drop_vif, axis=1, inplace=True)
Test_X.drop(cols_to_drop_vif, axis=1, inplace=True)



 
###############################################
# Model building with cutoff probability = 0.5
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.5, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.5, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 



###############################################
# Model building with cutoff probability = 0.3
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.3, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.3, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 


###############################################
# Model building with cutoff probability = 0.2
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.2, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.2, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 

###############################################
# Model building with cutoff probability = 0.1
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.1, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.1, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 


###############################################
# Model building with cutoff probability = 0.05
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.05, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.05, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 


###############################################
# Model building with cutoff probability = 0.08
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.08, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.08, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 

###############################################
# Model building with cutoff probability = 0.09
###############################################

# Build logistic regression model (using sklearn package/library)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
M1_Model = logreg.fit(Train_X,Train_Y)
predictions = M1_Model.predict_proba(Test_X)
predictions_1 = predictions[:,1]

# Classify 0 or 1 based on 0.5 cutoff
predict_class = np.where(predictions_1 >= 0.09, 1, 0)

# Confusion matrix
Confusion_Mat = pd.crosstab(predict_class, Test_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100


########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
prediction_train = M1_Model.predict_proba(Train_X)


predict_train_class = np.where(prediction_train[:,1] >= 0.09, 1, 0)

# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train_Y, predict_train_class)

# Plot ROC Curve
ROC_Df = pd.DataFrame()
ROC_Df['FPR'] = fpr 
ROC_Df['TPR'] = tpr
ROC_Df['Cutoff'] = thresholds

# Plot ROC Curve
import matplotlib.pyplot as plt
plt.plot(ROC_Df.FPR, ROC_Df.TPR) # (x,y)

# Area under curve (AUC)
auc(fpr, tpr)

from sklearn import metrics
metrics.accuracy_score(Test_Y, predict_class)
metrics.recall_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 
metrics.precision_score(Test_Y, predict_class) # very low score as meaning of 0 and 1 for default ind is opposite 


##########################################################################################
# Multicollinearity check after first round of model building and obtaining 
# cutoff values (0.5,0.3.0.2,0.1,0.05, 0.08, 0.09)
# against which we need to evaulate the model
##########################################################################################

# Check for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor


cols_to_drop_vif = []
# All columns with vif value > 10 were earmarked to be dropped from analysis
for i in range(Train_X.shape[1]-1):
    temp_vif = variance_inflation_factor(Train_X.values, i) # Pass Train_X.values and i (col_number)
    print(Train_X.columns[i], ": ", temp_vif)
    if(temp_vif>10):
        print('Since vif value is greater than 10 so dropping the column ',Train_X.columns[i])
        cols_to_drop_vif.append(Train_X.columns[i])
    
Train_X.drop(cols_to_drop_vif, axis=1, inplace=True)
Test_X.drop(cols_to_drop_vif, axis=1, inplace=True)
