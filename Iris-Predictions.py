#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


dataset=pd.read_csv('Iris.csv')


# In[4]:


dataset.head()


# In[5]:


X=dataset.drop('Species',axis=1)
y=dataset['Species']


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.7)


# In[7]:


from sklearn import tree
classifier=tree.DecisionTreeClassifier()


# In[8]:


from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()


# In[9]:


classifier.fit(X_train,y_train)


# In[17]:


from sklearn.metrics import accuracy_score
predictions=classifier.predict(X_test)
print(accuracy_score(y_test,predictions))

