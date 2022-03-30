#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


# In[2]:


data_set=pd.read_csv("homeprices2.csv")


# In[3]:


data_set.head()


# In[4]:


x=data_set.drop(['price'],axis=1)
y=data_set.drop(['area'],axis=1)


# In[5]:


x


# In[6]:


y


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[9]:


x_train


# In[10]:


y_train


# In[11]:


from sklearn.linear_model import LinearRegression


# In[12]:


lr=LinearRegression()


# In[13]:


m=lr.fit(x_train,y_train)


# In[14]:


pickle.dump(lr,open('model.pkl','wb'))


# In[ ]:




