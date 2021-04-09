# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 16:59:58 2021

@author: Aben George

"""
# HR ANALYTICS - TASKS

# 1) Test Correlation of variables on retention
# 2) Build Regression Model
# 3) Test Accuracy

#---------------------------------------------------------

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# -----------------------------------------------------

df = pd.read_csv('HR_comma_sep.csv')

# 1) Create two df one for left, one for retained

left = df[df.left==1]
left.shape

retained = df[df.left==0]
retained.shape

# M1 to check correlation - check mean diff width 

print(df.groupby('left').mean())

corr = df.groupby('left').mean() # assigned since print was not getting all values

#Salary, Avg month hours & Promotion in last 5 years shows higher mean diff that

# correlates to employee leaving



# CHECK CORRELATION OF CAT(salary and department) USING CROSSTAB GRAPH 

pd.crosstab(df.salary,df.left).plot(kind='bar')

pd.crosstab(df.Department,df.left).plot(kind='bar')

# Salary has high vaariation on left, Department is sort of consistent

# BELOW ARE THE VARIABLES WITH HIGHEST CORR 

# satisfaction_level, salary, average_montly_hours, promotion_last_5years

subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]

#-------------------------------------

# Creating a log reg model based off these variable


# NEED TO CONVERT SALARY & DEPARTMENT CATEGORICAL 

# M1 | USING CAT.CODES + ISSUE

# assigns unique integer values to each category variable
# using Cat.Codes will work better with ORDINAL CATEGORIES
            # ex- passenger number

# salary and department are not ordinal, hence cat values here would
#               lead to poor model performance



#df['salary'] = df['salary'].astype('category')
#df['sal_cat'] = df['salary'].cat.codes


#df['Department'] = df['Department'].astype('category')
#df['dep_cat'] = df['Department'].cat.codes

#------------------------------------------------------------------

# M2 | USING PD.GET_DUMMIES()

#   assigns cat values using a matrix
# GOOD FOR UNORDINAL DATA

# Make dummies, assign into df, remove original cat values

salary_dummies = pd.get_dummies(subdf.salary, prefix="salary")
df_with_dummies = pd.concat([subdf,salary_dummies],axis='columns')
df_with_dummies.drop('salary',axis='columns',inplace=True)
#----------------------------------------------------------------------

# 2) CREATE MODEL
# X = Independant variables in numeric datafram
# y = dependant variable

X = df_with_dummies
y = df.left

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.3)


model = LogisticRegression()

model.fit(X_train, y_train)

pred_df=model.predict(X_test)

print(model.predict(X_test)) # HERE 0 = LEFT, 1 = STAYED

#--------------------------------------------------------------------------

# 3) ACCURACY 

print(model.score(X_test,y_test))

