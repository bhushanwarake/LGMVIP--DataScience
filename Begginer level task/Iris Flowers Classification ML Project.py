#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv('IRIS.csv')


# In[3]:


data.head()


# In[4]:


# Giving Proper Heading to Columns
data_header = ['SepalLength','SepalWidth','PetalLength','PetalWidth','Species']
data.to_csv('Iris.csv', header = data_header, index = False)
new_data = pd.read_csv('Iris.csv')
new_data.head()


# In[5]:


# Checking no. of rows and columns
new_data.shape


# In[6]:


# Checking datatypes in dataset
new_data.info()


# In[7]:


# Describing Dataset
new_data.describe()


# In[8]:


# Checking Null Values in DataSet
new_data.isnull().sum()


# In[9]:


# Sepal Length vs Type
plt.bar(new_data['Species'],new_data['SepalLength'], width = 0.5) 
plt.title("Sepal Length vs Type")
plt.show()


# In[10]:


# Sepal Width vs Type
plt.bar(new_data['Species'],new_data['SepalWidth'], width = 0.5) 
plt.title("Sepal Width vs Type")
plt.show()


# In[11]:


# Petal Length vs Type
plt.bar(new_data['Species'],new_data['PetalLength'], width = 0.5) 
plt.title("Petal Length vs Type")
plt.show()


# In[12]:


# Petal Width vs Type
plt.bar(new_data['Species'],new_data['PetalWidth'], width = 0.5) 
plt.title("Petal Width vs Type")
plt.show()


# In[13]:


sns.pairplot(new_data,hue='Species')


# In[14]:


x = new_data.drop(columns="Species")
y = new_data["Species"]


# In[15]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)


# In[16]:


x_train.head()


# In[17]:


x_test.head()


# In[18]:


y_train.head()


# In[19]:


print("x_train: ", len(x_train))
print("x_test: ", len(x_test))
print("y_train: ", len(y_train))
print("y_test: ", len(y_test))


# In[20]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score


# In[21]:


model = LogisticRegression()
model.fit(x_train, y_train)


# In[22]:


predict = model.predict(x_test)
print("Pridicted values on Test Data", predict)


# In[23]:


y_test_pred = model.predict(x_test)
y_train_pred = model.predict(x_train)


# In[24]:


print("Training Accuracy : ", accuracy_score(y_train, y_train_pred))
print("Test Accuracy : ", accuracy_score(y_test, y_test_pred))


# In[ ]:




