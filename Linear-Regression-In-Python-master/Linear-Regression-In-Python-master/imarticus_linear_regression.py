#Linear regression
# Set working directory
#path = input("Input file path directory: ")
import os
#os.chdir(path)
os.chdir("D:")
os.getcwd()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
plt.style.use('bmh')

# Reading complete dataset
dataset = pd.read_csv('R_Module_Day_5.2_Data_Case_Study_Loss_Given_Default.csv',sep = ',',low_memory=False)

from sklearn.model_selection import train_test_split
dataset_train ,dataset_test = train_test_split(dataset,test_size=0.2)
dataset_train['source'] = 'Train'
dataset_test['source'] = 'Test'


# Combining the dataset again for further computations
dataset_full = pd.concat([dataset_train,dataset_test], axis = 0)

#del [dataset, dataset_train,dataset_test]

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
## Outlier Detection & Correction 
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
dataset_full.drop(['Ac_No'], axis=1, inplace=True)


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

dataset_full2['Age'].min()
dataset_full2['Age'].max()

dataset_full2['Years of Experience'].min()
dataset_full2['Years of Experience'].max()

# Creating Interaction variables

# Age_bin
    
bins = [10,20,30,40,50,60,70]
dataset_full2['Age_bin'] = pd.cut(dataset_full2['Age'], bins).astype(str)

# YoE_bin
bins = [-1,10,20,30,40,50,60]
dataset_full2['YoE_bin'] = pd.cut(dataset_full2['Years of Experience'], bins).astype(str)


# Correcting the bin (-1, 10] to (0, 10]
dataset_full2.YoE_bin = dataset_full2.YoE_bin.replace({"(-1, 10]": "(0, 10]"})
  

    
dataset_full2.drop(['Age','Years of Experience'], axis=1, inplace=True)

dataset_full2.dtypes
    

#########################################
## Feature Engineering Section
#########################################


# Creating interaction variables with below combination to add more features to data set:

#Age_bin
#YoE_bin
#Married_Single
#Gender_M
#Number of Vehicles 

Age_bin_crosstab_veh = pd.crosstab(dataset_full2['Age_bin'], dataset_full2['Number of Vehicles'], normalize='index')
YoE_bin_crosstab_veh = pd.crosstab(dataset_full2['YoE_bin'], dataset_full2['Number of Vehicles'], normalize='index')

Age_bin_crosstab_gen = pd.crosstab(dataset_full2['Age_bin'], dataset_full2['Gender_M'], normalize='index')
YoE_bin_crosstab_gen = pd.crosstab(dataset_full2['YoE_bin'], dataset_full2['Gender_M'], normalize='index')

Age_bin_crosstab_marr = pd.crosstab(dataset_full2['Age_bin'], dataset_full2['Married_Single'], normalize='index')
YoE_bin_crosstab_marr = pd.crosstab(dataset_full2['YoE_bin'], dataset_full2['Married_Single'], normalize='index')

# Age_bin & Gender_M
Age_bin_crosstab_gen['index1'] = Age_bin_crosstab_gen.index
for i in Age_bin_crosstab_gen['index1']:
    aa_0 = Age_bin_crosstab_gen.loc[Age_bin_crosstab_gen.index1==i,][0].round().astype(str)
    aa_1 = Age_bin_crosstab_gen.loc[Age_bin_crosstab_gen.index1==i,][1].round().astype(str)
    #print(i,"    ",aa_0,"    "+aa_1)
    dataset_full2['Age_bin_Gender_M'] = np.where(dataset_full2['Age_bin']==i,dataset_full2['Age_bin']+"_"+aa_0[0],dataset_full2['Age_bin']+"_"+aa_1[0])

# Age_bin & Married_Single
Age_bin_crosstab_marr['index1'] = Age_bin_crosstab_marr.index
for i in Age_bin_crosstab_marr['index1']:
    aa_0 = Age_bin_crosstab_marr.loc[Age_bin_crosstab_marr.index1==i,][0].round().astype(str)
    aa_1 = Age_bin_crosstab_marr.loc[Age_bin_crosstab_marr.index1==i,][1].round().astype(str)
    #print(i,"    ",aa_0,"    "+aa_1)
    dataset_full2['Age_bin_Married_Single'] = np.where(dataset_full2['Age_bin']==i,dataset_full2['Age_bin']+"_"+aa_0[0],dataset_full2['Age_bin']+"_"+aa_1[0])

# YoE_bin & Gender_M
YoE_bin_crosstab_gen['index1'] = YoE_bin_crosstab_gen.index
for i in YoE_bin_crosstab_gen['index1']:
    aa_0 = YoE_bin_crosstab_gen.loc[YoE_bin_crosstab_gen.index1==i,][0].round().astype(str)
    aa_1 = YoE_bin_crosstab_gen.loc[YoE_bin_crosstab_gen.index1==i,][1].round().astype(str)
    #print(i,"    ",aa_0,"    "+aa_1)
    dataset_full2['YoE_bin_Gender_M'] = np.where(dataset_full2['YoE_bin']==i,dataset_full2['YoE_bin']+"_"+aa_0[0],dataset_full2['YoE_bin']+"_"+aa_1[0])

# YoE_bin & Married_Single
YoE_bin_crosstab_marr['index1'] = YoE_bin_crosstab_marr.index
for i in YoE_bin_crosstab_marr['index1']:
    aa_0 = YoE_bin_crosstab_marr.loc[YoE_bin_crosstab_marr.index1==i,][0].round().astype(str)
    aa_1 = YoE_bin_crosstab_marr.loc[YoE_bin_crosstab_marr.index1==i,][1].round().astype(str)
    #print(i,"    ",aa_0,"    "+aa_1)
    dataset_full2['YoE_bin_Married_Single'] = np.where(dataset_full2['YoE_bin']==i,dataset_full2['YoE_bin']+"_"+aa_0[0],dataset_full2['YoE_bin']+"_"+aa_1[0])


#aa_1 = Age_bin_crosstab_veh.loc[Age_bin_crosstab_veh.index1=='(10, 20]',][1.0].round(1).astype(str)

# Age_bin & Number of Vehicles
Age_bin_crosstab_veh['index1'] = Age_bin_crosstab_veh.index
for i in Age_bin_crosstab_veh['index1']:
    aa_1 = Age_bin_crosstab_veh.loc[Age_bin_crosstab_veh.index1==i,][1.0].round(1).astype(str)
    aa_2 = Age_bin_crosstab_veh.loc[Age_bin_crosstab_veh.index1==i,][2.0].round(1).astype(str)
    aa_3 = Age_bin_crosstab_veh.loc[Age_bin_crosstab_veh.index1==i,][3.0].round(1).astype(str)
    aa_4 = Age_bin_crosstab_veh.loc[Age_bin_crosstab_veh.index1==i,][4.0].round(1).astype(str)
    
    dataset_full2.loc[(dataset_full2['Age_bin']==i) & (dataset_full2['Number of Vehicles']==1),'Age_bin_Vehicles'] = dataset_full2['Age_bin']+"_1_"+aa_1[0]
    dataset_full2.loc[(dataset_full2['Age_bin']==i) & (dataset_full2['Number of Vehicles']==2),'Age_bin_Vehicles'] = dataset_full2['Age_bin']+"_2_"+aa_2[0]
    dataset_full2.loc[(dataset_full2['Age_bin']==i) & (dataset_full2['Number of Vehicles']==3),'Age_bin_Vehicles'] = dataset_full2['Age_bin']+"_3_"+aa_3[0]
    dataset_full2.loc[(dataset_full2['Age_bin']==i) & (dataset_full2['Number of Vehicles']==4),'Age_bin_Vehicles'] = dataset_full2['Age_bin']+"_4_"+aa_4[0]






# YoE_bin & Number of Vehicles
YoE_bin_crosstab_veh['index1'] = YoE_bin_crosstab_veh.index
for i in YoE_bin_crosstab_veh['index1']:
    aa_1 = YoE_bin_crosstab_veh.loc[YoE_bin_crosstab_veh.index1==i,][1.0].round(1).astype(str)
    aa_2 = YoE_bin_crosstab_veh.loc[YoE_bin_crosstab_veh.index1==i,][2.0].round(1).astype(str)
    aa_3 = YoE_bin_crosstab_veh.loc[YoE_bin_crosstab_veh.index1==i,][3.0].round(1).astype(str)
    aa_4 = YoE_bin_crosstab_veh.loc[YoE_bin_crosstab_veh.index1==i,][4.0].round(1).astype(str)
    
    dataset_full2.loc[(dataset_full2['YoE_bin']==i) & (dataset_full2['Number of Vehicles']==1),'YoE_bin_Vehicles'] = dataset_full2['YoE_bin']+"_1_"+aa_1[0]
    dataset_full2.loc[(dataset_full2['YoE_bin']==i) & (dataset_full2['Number of Vehicles']==2),'YoE_bin_Vehicles'] = dataset_full2['YoE_bin']+"_2_"+aa_2[0]
    dataset_full2.loc[(dataset_full2['YoE_bin']==i) & (dataset_full2['Number of Vehicles']==3),'YoE_bin_Vehicles'] = dataset_full2['YoE_bin']+"_3_"+aa_3[0]
    dataset_full2.loc[(dataset_full2['YoE_bin']==i) & (dataset_full2['Number of Vehicles']==4),'YoE_bin_Vehicles'] = dataset_full2['YoE_bin']+"_4_"+aa_4[0]




###########################################################################
# Dummy variable creation
# From above step we come to know that there is no category ratinalization 
# so we can directly hop onto dummy creation step
###########################################################################       
        
# Create dummy variables from all indep categorical variables

# Step 1: Identify categorical vars
Categ_Vars1 = dataset_full2.loc[:,dataset_full2.dtypes == object].columns

# Step 2: Create dummy vars
Dummy_Df1 = pd.get_dummies(dataset_full2[Categ_Vars1].drop(['source'], axis = 1), drop_first=True, dtype=int)
Dummy_Df1.columns
Dummy_Df1.shape
Dummy_Df1.dtypes

# Step 3: Append the Dummy_Df with dataset_full. Call it dataset_full2
dataset_full3 = pd.concat([dataset_full2, Dummy_Df1], axis = 1)
dataset_full3.shape
dataset_full3.dtypes

# Step 4.1: Drop all the irrelavant and categorical columns (Do NOT drop Source column - We need it for sample splitting)
Cols_To_Drop1 = Categ_Vars1.drop('source') # Ensure you discard 'Source' column from "columns to drop"

# Step 4.2
dataset_full3.drop(Cols_To_Drop1, axis=1, inplace=True)
dataset_full3.shape
dataset_full3.columns
    

########################
# Sampling
########################

# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column
Train = dataset_full3.loc[dataset_full3.source == "Train",:].drop('source', axis = 1).copy()
Train.shape
Test = dataset_full3.loc[dataset_full3.source == "Test",:].drop('source', axis = 1).copy()
Test.shape

for column in Train:
    if(Train[column].nunique()==1):
        Train.drop(column, axis=1, inplace=True)
        Test.drop(column, axis=1, inplace=True)   
        

########################
# Divide data into Train_X, Train_Y & Test_X, Test_Y
########################

# Divide each dataset into Indep Vars and Dep var
Train_X = Train.drop('Losses in Thousands', axis = 1).copy()
Train_Y = Train['Losses in Thousands'].copy()
Test_X = Test.drop('Losses in Thousands', axis = 1).copy()
Test_Y = Test['Losses in Thousands'].copy()

Train_X.shape


########################
# Multicollinearity check
########################

# Check for VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

#cols_to_drop_vif=[]
#for i in range(Train_X.shape[1]-1):
#    temp_vif = variance_inflation_factor(Train_X.values, i) # Pass Train_X.values and i (col_number)
#    print(Train_X.columns[i], ": ", temp_vif)
#    if(temp_vif>10):
#        print('Since vif value is greater than 10 so dropping the column ',temp_vif, Train_X.columns[i])
#        cols_to_drop_vif.append(Train_X.columns[i])
#        #Train_X.drop(Train_X.columns[i], axis=1, inplace=True)
#        #Test_X.drop(Test_X.columns[i], axis=1, inplace=True)


cols_to_drop_vif = pd.DataFrame(columns=['column_name','vif_value'])
cols_to_drop_vif['vif_value'] = cols_to_drop_vif.vif_value.astype(float)
cols_to_drop_vif.dtypes
cols_to_drop_inf = []
import math

for i in range(Train_X.shape[1]-1):
    if math.isinf(variance_inflation_factor(Train_X.values, i)):
        cols_to_drop_inf.append(Train_X.columns[i])
    elif (variance_inflation_factor(Train_X.values, i)>10):
        cols_to_drop_vif['vif_value'] = variance_inflation_factor(Train_X.values, i)
        cols_to_drop_vif['column_name'] = Train_X.columns[i]


Train_X.drop(cols_to_drop_inf, axis=1, inplace=True)
Test_X.drop(cols_to_drop_inf, axis=1, inplace=True)

### In this example we do not get the situation where cols_to_drop_vif dataframne is populated
### otherwise we will have to iterate through that and remove column with maximum vif value

Train_X.dtypes
Train_X['YoE_bin_Vehicles_(50, 60]_4_0.2'].nunique()

########################
# Model building
########################

# Build linear regression model (using sklearn package/library)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
M1_Model = regressor.fit(Train_X, Train_Y)

import statsmodels.api as sm

Train_X1 = sm.add_constant(Train_X)
regressor_OLS = sm.OLS(endog = Train_Y, exog = Train_X1).fit() 
regressor_OLS.summary()

Train_X.drop('Number of Vehicles', axis=1, inplace=True)
Test_X.drop('Number of Vehicles', axis=1, inplace=True)

Train_X2 = sm.add_constant(Train_X)
regressor_OLS = sm.OLS(endog = Train_Y, exog = Train_X2).fit() 
regressor_OLS.summary()

M2_Model = regressor.fit(Train_X, Train_Y)

# Predicting the test set results
y_pred = M1_Model.predict(Test_X)

# MAPE Calculation and testing
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(Test_Y, y_pred)
