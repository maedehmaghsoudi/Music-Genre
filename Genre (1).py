#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data=pd.read_csv('music_dataset_mod.csv')


# In[3]:


data


# In[4]:


test=data[data['Genre'].isnull()]
test


# In[5]:


y_test=test['Genre']
y_test
x_test=test.drop('Genre',axis=1)
x_test


# In[6]:


train=data.dropna()
train


# In[7]:


x_train= train.drop('Genre', axis=1)
x_train


# In[8]:


y_train=train['Genre']
y_train


# In[9]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_train_scaled


# In[10]:


#chera x.corr javab darad, x_scaled.corr na? ghable scale corr o bezan


# In[11]:


from sklearn.decomposition import PCA


# In[12]:


pca = PCA(n_components=0.8)


# In[13]:


x_train_scaled_pca=pca.fit_transform(x_train_scaled)
x_train_scaled_pca


# In[14]:


from sklearn.linear_model import LogisticRegression


# In[15]:


model=LogisticRegression()


# In[16]:


model.fit(x_train_scaled_pca,y_train)


# In[17]:


x_test_scaled= scaler.transform(x_test)
x_test_scaled_pca=pca.transform(x_test_scaled)
x_test_scaled_pca


# In[19]:


y_test=model.predict(x_test_scaled_pca)


# In[20]:


y_test


# In[ ]:


#10% train joda mikonam , be onvane validation. 


# In[ ]:


#we get the coeff for interpreting the case. we need to get back to from pca then scaler. 


# In[29]:


model.coef_


# In[30]:


model.coef_.shape


# In[31]:


pca.components_


# In[28]:


pca.components_.shape


# In[33]:


org_coeff_pca=np.dot(model.coef_,pca.components_)


# In[34]:


org_coeff_pca


# In[35]:


org_coeff_pca_scaler=org_coeff_pca/scaler.scale_


# In[36]:


org_coeff_pca_scaler


# In[ ]:


#in 5 ta hamoon 5 ta genre hastan ke zaribe har kodoom az moteghayera ro neshoon midan. va tasire har kodom.


# In[ ]:


#most people is important for them to see accuracy in model but interpertation is imortsnt for companies.


# In[37]:


from sklearn.model_selection import train_test_split


# In[ ]:


#accuracy ham roo train ham roo validation begirim.confusion matrix ham begiram tafsiresh jalebe.


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




