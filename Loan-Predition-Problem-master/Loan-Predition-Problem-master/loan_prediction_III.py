# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 18:30:49 2018

@author: Gaurav Rai
"""

# Set working directory
#path = input("Input file path directory: ")
import os
#os.chdir(path)
os.chdir('D:\Analytics_Vidhya_Research\Loan Prediction Problem')
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')

# Reading complete dataset
dataset = pd.read_csv('train_u6lujuX_CVtuZ9i.csv',sep = ',',low_memory=False)

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
dataset_full.drop(['Loan_ID'], axis=1, inplace=True)

# Encoding the loan_status column and removing the loan_status column from the dataset_full dataframe
dataset_full['Loan_Status_Y'] = np.where(dataset_full.Loan_Status == 'Y',1,0)
dataset_full.drop(['Loan_Status'], axis=1, inplace=True)

#####################################################################
## Creating bins using pd.crosstabs and looking closely at the data
##################################################################### 
for column in dataset_full:
    if(dataset_full[column].dtype == 'object'):
        
        print('before >>>>> ',column,'::::',dataset_full[column].nunique())
        a = pd.crosstab(dataset_full[column], dataset_full['Loan_Status_Y'], normalize='index')

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
Train_X = Train.drop('Loan_Status_Y', axis = 1).copy()
Train_Y = Train['Loan_Status_Y'].copy()
Test_X = Test.drop('Loan_Status_Y', axis = 1).copy()
Test_Y = Test['Loan_Status_Y'].copy()


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
# Model building
########################

# Build logistic regression model (using statsmodels package/library)
import statsmodels.api as sm
M1 = sm.Logit(Train_Y, Train_X) # (Dep_Var, Indep_Vars) # This is model definition
M1_Model = M1.fit() # This is model building
M1_Model.summary() # This is model output summary


#################################################
# Manual model selection. 
# Drop the most insignificant variable in model 
# one by one and recreate the model
# variable with p-score>0.05 is insignificant
################################################

# Drop CoapplicantIncome as its p-score is highest
Cols_To_Drop = ['CoapplicantIncome']
M2 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M2_Model = M2.fit()
M2_Model.summary()

# Drop Dependents_2 as its p-score is highest
Cols_To_Drop.append('Dependents_2')
M3 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M3_Model = M3.fit()
M3_Model.summary()


# Drop Gender_Male as its p-score is highest
Cols_To_Drop.append('Gender_Male')
M4 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M4_Model = M4.fit()
M4_Model.summary()


# Drop Dependents_3+ as its p-score is highest
Cols_To_Drop.append('Dependents_3+')
M5 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M5_Model = M5.fit()
M5_Model.summary()

# Drop Self_Employed_Yes as its p-score is highest
Cols_To_Drop.append('Self_Employed_Yes')
M6 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M6_Model = M6.fit()
M6_Model.summary()

# Drop ApplicantIncome as its p-score is highest
Cols_To_Drop.append('ApplicantIncome')
M7 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M7_Model = M7.fit()
M7_Model.summary()


# Drop Property_Area_Urban as its p-score is highest
Cols_To_Drop.append('Property_Area_Urban')
M8 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M8_Model = M8.fit()
M8_Model.summary()

# Drop Dependents_1 as its p-score is highest
Cols_To_Drop.append('Dependents_1')
M9 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M9_Model = M9.fit()
M9_Model.summary()

# Drop Dependents_1 as its p-score is highest
Cols_To_Drop.append('LoanAmount')
M10 = sm.Logit(Train_Y, Train_X.drop(Cols_To_Drop, axis = 1)) # (Dep_Var, Indep_Vars)
M10_Model = M10.fit()
M10_Model.summary()

########################
# Predict on testset
########################

Columns_To_Use = Train_X.drop(Cols_To_Drop, axis = 1).columns # Indentify important columns from modeling
Test['Test_Prob'] = M10_Model.predict(Test[Columns_To_Use]) # Use Test to store the predicted probs
Test.head()

# Classify 0 or 1 based on 0.5 cutoff
# Checked for values > 0.5 and < 0.5 and best result was obtained at 0.5 and 
# maintaining universal rule that for binary classifier the best case event 
#probabilty is 0.5
import numpy as np
Test['Test_Class'] = np.where(Test.Test_Prob >= 0.5, 1, 0)
Test.columns

# Confusion matrix
Confusion_Mat = pd.crosstab(Test.Test_Class, Test.Loan_Status_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100

## Answer is accuracy = 82.11%

#############################
# Predict on actual testset
#############################
test_final = pd.read_csv('test_Y3wMUE5_7gLdaTN.csv',sep = ',',low_memory=False)


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
        
        
# Dropping the Loan ID key fields as it is just for identification and does not add any value to the analysis
test_final_1 = pd.DataFrame()
test_final_1['Loan_ID'] = test_final['Loan_ID']
test_final.drop(['Loan_ID'], axis=1, inplace=True)

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




test_final2['Test_Prob'] = M10_Model.predict(test_final2[Columns_To_Use]) # Use Test to store the predicted probs
test_final2.head()

# Classify 0 or 1 based on 0.5 cutoff
# Checked for values > 0.5 and < 0.5 and best result was obtained at 0.5 and 
# maintaining universal rule that for binary classifier the best case event 
#probabilty is 0.5
import numpy as np
test_final2['Test_Class'] = np.where(test_final2.Test_Prob >= 0.5, 1, 0)
test_final2.columns

# Confusion matrix
Confusion_Mat = pd.crosstab(test_final2.Test_Class, test_final2.Loan_Status_Y) # R, C format
Confusion_Mat

Confusion_Mat[0]
Confusion_Mat[0][0]

# Check the accuracy of the model
((Confusion_Mat[0][0] + Confusion_Mat[1][1])/Test.shape[0])*100



## Final Answer is accuracy on actual test data = 73.17%
