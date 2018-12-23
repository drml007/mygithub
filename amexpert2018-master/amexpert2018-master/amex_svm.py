# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 20:45:37 2018

@author: Gaurav Rai
"""

# Set working directory
#path = input("Input file path directory: ")
import os
#os.chdir(path)
os.chdir('D:\Analytics_Vidhya_Research\AM_Expert_2018 Competition')
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')

# Reading complete dataset
dataset = pd.read_csv('train.csv',sep = ',',low_memory=False)

# Check the data types of the columns
data_dtypes = dataset.dtypes

from sklearn.model_selection import train_test_split
dataset_train ,dataset_test = train_test_split(dataset,test_size=0.2)
dataset_train['source'] = 'Train'
dataset_test['source'] = 'Test'


# Combining the dataset again for further computations
dataset_full = pd.concat([dataset_train,dataset_test], axis = 0)

########################
# Missing value imputation (NAs)
########################

# First check for NAs
# returns the sum of NA values column wise
b = dataset_full.isnull().sum() 

# For Continuous vars: Impute with median
# For Categorical vars: Impute with mode

for column in dataset_full:
    if( (dataset_full[column].dtype == 'object') & (b[column] > 0)):
        dataset_full[column].fillna(dataset_train[column].mode()[0], inplace = True)
    elif ((dataset_full[column].dtype != 'object') & (b[column] > 0)):
        dataset_full[column].fillna(dataset_train[column].median(), inplace = True)

c = dataset_full.isnull().sum() 



###########################################
## Outlier Detection & Correction - to do 
########################################### 

for column in dataset_train:
    if (dataset_train[column].dtype != 'object'):
        q = dataset_train[column].quantile(0.99)
        r = dataset_train[column].quantile(0.01)
        dataset_full[column] = np.where(dataset_full[column] > q, q, dataset_full[column])
        dataset_full[column] = np.where(dataset_full[column] < r, r, dataset_full[column])
            
            
# In this dataset there are certain columns which contain all 
# nulls (in full dataset or train ) so we need to remove all those columns
for i in c.index:
    if(c[i]>0):
        dataset_full.drop(i, axis=1, inplace=True)
        
        
# Dropping the Loan ID key fields as it is just for identification and does not add any value to the analysis
dataset_full.drop(['session_id','DateTime'], axis=1, inplace=True)


#####################################################################
## Creating bins using pd.crosstabs and looking closely at the data
##################################################################### 
for column in dataset_full:
    if(dataset_full[column].dtype == 'object'):
        
        print('before >>>>> ',column,'::::',dataset_full[column].nunique())
        a = pd.crosstab(dataset_full[column], dataset_full['is_click'], normalize='index')

        temp_desc = a.index[a[0] == 0]
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_0",dataset_full[column])            
            
        
        temp_desc = a.index[a[0] ==1] 
        dataset_full[column] = np.where(dataset_full[column].isin(temp_desc),column+"_1",dataset_full[column])           

        print('after >>>>> ',column,'::::',dataset_full[column].nunique())
        
        
###########################################################################
# Dummy variable creation
# From above step we come to know that there is no category ratinalization 
# so we can directly hop onto dummy creation step
###########################################################################       
        
# Create dummy variables from all indep categorical variables

# Step 1: Identify categorical vars
Categ_Vars = dataset_full.loc[:,dataset_full.dtypes == object].columns

# Step 2: Create dummy vars
Dummy_Df = pd.get_dummies(dataset_full[Categ_Vars].drop(['source'], axis = 1), drop_first=True, dtype=int)
Dummy_Df.columns
Dummy_Df.shape
Dummy_Df.dtypes

# Step 3: Append the Dummy_Df with dataset_full. Call it dataset_full2
dataset_full2 = pd.concat([dataset_full, Dummy_Df], axis = 1)
dataset_full2.shape
dataset_full2.dtypes

# Step 4.1: Drop all the irrelavant and categorical columns (Do NOT drop Source column - We need it for sample splitting)
Cols_To_Drop = Categ_Vars.drop('source') # Ensure you discard 'Source' column from "columns to drop"

# Step 4.2
dataset_full2.drop(Cols_To_Drop, axis=1, inplace=True)
dataset_full2.shape
dataset_full2.columns


########################
# Sampling
########################

# intercept = 1 always gets multiplied at the backend and for easier 
# explanability we give value of 1 as its multiplication is easy
dataset_full2['Intercept'] = 1

# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column
Train = dataset_full2.loc[dataset_full2.source == "Train",:].drop('source', axis = 1).copy()
Train.shape
Test = dataset_full2.loc[dataset_full2.source == "Test",:].drop('source', axis = 1).copy()
Test.shape

#####################################################
# Divide data into Train_X, Train_Y & Test_X, Test_Y
#####################################################

# Divide each dataset into Indep Vars and Dep var
Train_X = Train.drop('is_click', axis = 1).copy()
Train_Y = Train['is_click'].copy()
Test_X = Test.drop('is_click', axis = 1).copy()
Test_Y = Test['is_click'].copy()


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



########################
# Feature Scaling
########################
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
Train_X = sc.fit_transform(Train_X)
Test_X = sc.fit_transform(Test_X)

########################
# Model building
########################


## Build a SVM model

from sklearn.svm import SVC
from sklearn.metrics import classification_report


M1 = SVC() # Takes a very long time when used with kernel = 'linear' : M1 = SVC(kernel = 'linear')

M1_Model = M1.fit(Train_X,Train_Y)
Test_Pred = M1_Model.predict(Test_X)

from sklearn.metrics import confusion_matrix
confusion_mat = confusion_matrix(Test_Y,Test_Pred)
confusion_mat

((confusion_mat[0][0] + confusion_mat[1][1])/Test_Y.shape[0])*100


## classification Model Validation
Report = classification_report(Test_Y,Test_Pred)
print(Report) ## use print for formatted reports


# SVM grid search code piece that did not work
from sklearn.model_selection import GridSearchCV

my_param_grid = {'C':[1,2],'gamma':[0.01,0.1],'kernel':['sigmoid','rbf']}

SVM_GS = GridSearchCV(SVC(), param_grid=my_param_grid, scoring = 'accuracy', cv = 10)

SVM_GS_Model = SVM_GS.fit(Train_X,Train_Y)
SVM_GS_Model.cv_results_

SVM_Grid_Search_DF = pd.DataFrame.from_dict(SVM_GS_Model.cv_results_)
SVM_GS_Model.best_params_


########################
# Predict on testset
########################

Columns_To_Use = Train_X.drop(Cols_To_Drop, axis = 1).columns # Indentify important columns from modeling
Test['Test_Prob'] = M4_Model.predict(Test[Columns_To_Use]) # Use Test to store the predicted probs
Test.head()


# Classify 0 or 1 based on 0.5 cutoff
# Checked for values > 0.5 and < 0.5 and best result was obtained at 0.5 and 
# maintaining universal rule that for binary classifier the best case event 
#probabilty is 0.5
import numpy as np
Test['Test_Class'] = np.where(Test.Test_Prob >= 0.091, 1, 0)
Test.columns

# Confusion matrix
Confusion_Mat = pd.crosstab(Test.Test_Class, Test.is_click) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100

## Answer is accuracy = 92.72%

########################
# AUC and ROC Curve
########################

from sklearn.metrics import roc_curve, auc

# Predict on train data
Train['Train_Prob'] = M4_Model.predict(Train[Columns_To_Use])


# Calculate FPR, TPR and Cutoff Thresholds
fpr, tpr, thresholds = roc_curve(Train['is_click'], Train['Train_Prob'])


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


#############################
# Predict on actual testset
#############################
test_final = pd.read_csv('test.csv',sep = ',',low_memory=False)

# Dropping the Loan ID key fields as it is just for identification and does not add any value to the analysis
test_final_1 = pd.DataFrame()
test_final_1['session_id'] = test_final['session_id']
test_final.drop(['session_id','DateTime'], axis=1, inplace=True)

##************************
##************************
########################
# Missing value imputation (NAs)
########################

# First check for NAs
# returns the sum of NA values column wise
b = test_final.isnull().sum() 

# For Continuous vars: Impute with median
# For Categorical vars: Impute with mode

for column in test_final:
    if( (test_final[column].dtype == 'object') & (b[column] > 0)):
        test_final[column].fillna(test_final[column].mode()[0], inplace = True)
    elif ((test_final[column].dtype != 'object') & (b[column] > 0)):
        test_final[column].fillna(test_final[column].median(), inplace = True)

c = test_final.isnull().sum() 

###########################################
## Outlier Detection & Correction - to do 
########################################### 

for column in test_final:
    if (test_final[column].dtype != 'object'):
        q = test_final[column].quantile(0.99)
        r = test_final[column].quantile(0.01)
        test_final[column] = np.where(test_final[column] > q, q, test_final[column])
        test_final[column] = np.where(test_final[column] < r, r, test_final[column])
            
            
# In this dataset there are certain columns which contain all 
# nulls (in full dataset or train ) so we need to remove all those columns
for i in c.index:
    if(c[i]>0):
        test_final.drop(i, axis=1, inplace=True)
        
        


###########################################################################
# Dummy variable creation
# From above step we come to know that there is no category ratinalization 
# so we can directly hop onto dummy creation step
###########################################################################       
        
# Create dummy variables from all indep categorical variables

# Step 1: Identify categorical vars
Categ_Vars = test_final.loc[:,dataset_full.dtypes == object].columns

# Step 2: Create dummy vars
Dummy_Df = pd.get_dummies(test_final[Categ_Vars], drop_first=True, dtype=int)
Dummy_Df.columns
Dummy_Df.shape
Dummy_Df.dtypes

# Step 3: Append the Dummy_Df with dataset_full. Call it dataset_full2
test_final2 = pd.concat([test_final, Dummy_Df], axis = 1)
test_final2.shape
test_final2.dtypes


# Step 4.2
test_final2.drop(Categ_Vars, axis=1, inplace=True)
test_final2.shape
test_final2.columns

########################
# Sampling
########################

# intercept = 1 always gets multiplied at the backend and for easier 
# explanability we give value of 1 as its multiplication is easy
test_final2['Intercept'] = 1
##************************
##************************




test_final2['Test_Prob'] = M4_Model.predict(test_final2[Columns_To_Use]) # Use Test to store the predicted probs
test_final2.head()

# Classify 0 or 1 based on 0.5 cutoff
# Checked for values > 0.5 and < 0.5 and best result was obtained at 0.5 and 
# maintaining universal rule that for binary classifier the best case event 
#probabilty is 0.5
import numpy as np
test_final2['is_click'] = np.where(test_final2.Test_Prob >= 0.091, 1, 0)

test_result = pd.concat([test_final_1['session_id'], test_final2['is_click']], axis = 1)

test_result.to_csv('submission.csv', index=False)
