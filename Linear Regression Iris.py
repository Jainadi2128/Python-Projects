#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn import datasets
import numpy as np
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt


# In[2]:


iris = datasets.load_iris()


# In[3]:


iris


# In[4]:


# Split into features and labels
X = iris.data
y = iris.target


# In[5]:


print(X.shape)
print(y.shape)


# In[6]:


X = pd.DataFrame(X)
X


# In[7]:


X.columns = iris.feature_names


# In[8]:


X


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)


# In[11]:


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[12]:


# Algorithm
l_reg = linear_model.LinearRegression()


# In[15]:


# train
model = l_reg.fit(X_train,y_train)
prediction = l_reg.predict(X_test)


# In[18]:


print('Predictions:', prediction)
print('R^2 value: ', l_reg.score(X,y))
print('coedd:', l_reg.coef_)
print('intercept:', l_reg.intercept_)


# In[ ]:




