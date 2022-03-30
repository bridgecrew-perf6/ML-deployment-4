#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# In[2]:


data=pd.read_csv('IRIS.csv')


# In[3]:


data


# In[7]:


lb=LabelEncoder()


# In[8]:


data['species']=lb.fit_transform(data['species'])


# In[9]:


data


# In[13]:


X=data.iloc[:,0:4]


# In[17]:


y=data[['species']]


# In[14]:


X


# In[18]:


y


# In[20]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[21]:


lr=LogisticRegression()


# In[29]:


model=lr.fit(X_train,y_train)


# In[ ]:





# In[30]:


model.score(X_test,y_test)


# In[31]:


predictions=model.predict(X_test)


# In[32]:


predictions


# In[33]:


pickle.dump(lr,open('model.pkl','wb'))


# In[ ]:




