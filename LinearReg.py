#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION
# A statistical modelling method known as linear regression is used to analyse the connection between a dependent variable and one or more independent variables. The relationship between the variables is assumed to be linear, which means that changes in the dependent variable will affect the independent variables immediately.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


url="http://bit.ly/w-data"
data= pd.read_csv(url)


# In[3]:


data.describe()


# In[4]:


data.head()


# In[5]:


sns.barplot(x=data['Hours'],y=data['Scores'])


# In[6]:


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[7]:


x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values


# In[9]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train,y_train)


# In[11]:


y_pred= regressor.predict(x_test)


# In[12]:


plt.scatter(x_train,y_train, color="red")
plt.plot(x_train,regressor.predict(x_train))
plt.title("Hours v/s Percentage")
plt.xlabel("Hours")
plt.ylabel("Percentage")
plt.show()


# In[13]:


plt.scatter(x_test,y_test, color="red")
plt.plot(x_test,y_pred)
plt.show()


# In[14]:


regressor.predict([[9.25]])


# In[15]:


from sklearn import metrics
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:


#Thankyou

