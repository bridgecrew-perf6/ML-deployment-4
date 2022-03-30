#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import pickle
import joblib
from sklearn.model_selection import train_test_split


# In[3]:


salary_dataset=pd.read_csv('salary new.csv')


# In[4]:


salary_dataset.head()


# In[5]:


x=salary_dataset.drop(['Salary'],axis=1)
y=salary_dataset.drop(['YearsExperience'],axis=1)


# In[6]:


x


# In[7]:


y


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.3)


# In[9]:


x_test


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


regressor=LinearRegression()


# In[12]:


m=regressor.fit(x_train,y_train)


# In[13]:


pickle.dump(regressor,open('model.pkl','wb'))


# In[ ]:




