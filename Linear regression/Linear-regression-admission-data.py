#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv('Admission_Predict_linear_regression.csv')
df.head()


# In[3]:


df.info()


# In[4]:


df.isnull().sum()


# In[6]:


x=df.drop(['Sr no','Chance of admit'],axis=1)
y=df['Chance of admit']


# In[10]:


#plotting
for i in x.columns:
    plt.scatter(x[i],y)
    plt.xlabel(i)
    plt.ylabel('chance of admit')
    plt.show()


# In[13]:


#splitting
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.33,random_state=100)


# In[14]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)


# In[15]:


#accuracy
from sklearn.metrics import r2_score
score=r2_score(model.predict(x_test),y_test)
score


# In[28]:


#save model
file=open('model_linear_regression.pkl','wb')
pickle.dump(model,file)


# In[32]:


#load the model
model1=pickle.load(open('model_linear_regression.pkl','rb'))
a=model1.predict([[300,150,5,5,5,10,1]])
a

