# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 10:46:46 2021

@author: Aben George
"""


import pandas as pd
from matplotlib import pyplot as plt


df = pd.read_csv('insurance_data.csv')

#plt.scatter(df['age'],df['bought_insurance'], marker = '+', color = 'red')

# USING TRAIN TEST METHOD TO SPLIT DATA 

# TRAIN ONE SET | TEST ON THE OTHER

#print(df.shape)  # 27 ROWS, 2 COLUMNS

# --------------------------------------------------

# 1) CREATE THE TRAIN-TEST SPLIT

from sklearn.model_selection import train_test_split

#train_test_split(x(multidimensional array),y,test size)

x_train, x_test, y_train, y_test = train_test_split(df['age'], df['bought_insurance'], test_size = 0.1)


# test size = 0.1 means 90% of data will be used to train the model, 10% 
# for testing 

# \ = continue code to next line

#-----------------------------------------------

# 2) LOGISTIC REGRESION

from sklearn.linear_model import LogisticRegression

# Create reg object

model = LogisticRegression()

# .fit add training parameters
import numpy as np

#print(x_train.shape)

# SPYDER ISSUE - ALL THE .PREDICT,.FIT NEED ARRAYS(NOT IN JUPYTER)
# Create arrays, then reshape them(based on the size) since rows and columns would be interchanged 



x_ar = np.array(x_train)
y_ar = np.array(y_train)
xt_ar = np.array(x_test)
yt_ar = np.array(y_test)



model.fit(x_ar.reshape(24,1),y_ar.reshape(24,1).ravel())

print(x_test)
print(model.predict(xt_ar.reshape(3,1)))

# compare with original values in DF
#-----------------------------------------------------------

# 3)  CHECK MODEL SCORE

print(model.score(xt_ar.reshape(3,1),yt_ar.reshape(3,1)))

#--------------------------------------------------------------

# 4)  CHECK PROBABILTY VALUES

# RETURNS PROB OF 0 THEN PROB OF 1
# IE PROB OF NOT BUYING, PROB OF BUYING

print(model.predict_proba(xt_ar.reshape(3,1)))

#----------------------------------------------------------------

# PREDICT FOR A RANDOM AGE

print(model.predict(np.array([45]).reshape(1,-1)))






