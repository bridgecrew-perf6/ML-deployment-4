#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier


# In[29]:


data=pd.read_csv('Social_Network_Ads.csv')


# In[30]:


data


# In[31]:


le=LabelEncoder()


# In[32]:


data['Gender']=le.fit_transform(data['Gender'])


# In[33]:


data


# In[34]:


x=data.drop(['User ID','Purchased'],axis=1)
y=data[['Purchased']]


# In[35]:


x


# In[36]:


y


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)


# In[38]:


lr=LogisticRegression()
clf = DecisionTreeClassifier()


# In[39]:


model=clf.fit(x_train,y_train)


# In[40]:


model.score(x_train,y_train)


# In[41]:


model.score(x_train,y_train)


# In[42]:


x_test


# In[43]:


y_test


# In[44]:


prediction=model.predict(x_test)


# In[45]:


print(prediction)


# In[46]:


prediction


# In[47]:


a=x_test.iloc[0]


# In[48]:


a


# In[49]:


a=[[0,46,22000]]


# In[50]:


model.predict(a)


# In[51]:


b=[[1,45,25000]]


# In[52]:


model.predict(b)


# In[53]:


pickle.dump(clf,open('model.pkl','wb'))


# In[ ]:





# In[ ]:




