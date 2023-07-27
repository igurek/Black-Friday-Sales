#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IMPORT LIBRARIES

import pandas as pd
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt 

from sklearn import model_selection 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

import plotly.express as px
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')


# In[46]:


dstrain = pd.read_csv ("DStrain.csv")


# In[47]:


dstrain.head


# In[48]:


dstest= pd.read_csv ("DStest.csv")


# In[49]:


dstest.head


# In[50]:


dstrain.describe().T


# In[51]:


dstest.describe().T


# In[52]:


##Merge train and test data / not using merge 
ds=dstrain.append(dstest)
ds.head()


# In[ ]:





# In[53]:


ds.info()


# In[54]:


ds.describe().T


# In[55]:


## I don't need the column called "User ID"  
ds.drop(['User_ID'],axis=1,inplace=True)


# In[56]:


ds.head()


# In[ ]:


## Categorical values needs to be numerical values 


# In[58]:


ds['Gender']=pd.get_dummies(ds['Gender'],drop_first=1)


# In[59]:


ds.head(10)


# In[60]:


ds['Age'].unique()


# In[61]:


#pd.get_dummies(ds['Age'],drop_first=True)
ds['Age']=ds['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[62]:


ds.head()


# In[63]:


ds_city=pd.get_dummies(ds['City_Category'],drop_first=True)


# In[65]:


ds_city.head()


# In[66]:


ds=pd.concat([ds,ds_city],axis=1)
ds.head()


# In[67]:


ds.drop('City_Category',axis=1,inplace=True)


# In[68]:


ds.head(10)


# In[ ]:


## Handling Missing Values / not the purchase value but definetely the product category 1 and 2


# In[69]:


ds.isnull().sum()


# In[70]:


ds['Product_Category_2'].unique()


# In[71]:


ds['Product_Category_2'].value_counts()


# In[72]:


ds['Product_Category_2'].mode()[0]


# In[73]:


ds['Product_Category_2']=ds['Product_Category_2'].fillna(ds['Product_Category_2'].mode()[0])


# In[75]:


ds['Product_Category_2'].isnull().sum()


# In[76]:


ds['Product_Category_3'].unique()


# In[77]:


ds['Product_Category_3'].value_counts()


# In[79]:


ds['Product_Category_3']=ds['Product_Category_3'].fillna(ds['Product_Category_3'].mode()[0])


# In[80]:


ds.head()


# In[81]:


ds['Stay_In_Current_City_Years'].unique()


# In[82]:


##replacing 4+ with 4
ds['Stay_In_Current_City_Years']=ds['Stay_In_Current_City_Years'].str.replace('+','')


# In[83]:


ds.head()


# In[84]:


ds.info()


# In[85]:


ds['Stay_In_Current_City_Years']=ds['Stay_In_Current_City_Years'].astype(int)
ds.info()


# In[86]:


ds['B']=ds['B'].astype(int)
ds['C']=ds['C'].astype(int)


# In[88]:


ds.info()


# In[ ]:


## VISUALIZING PART 


# In[89]:


sns.barplot('Age','Purchase',hue='Gender',data=ds)


# In[ ]:


ds['Gender'].unique()


# In[94]:


sns.histplot(x='Purchase', data=ds, kde=True, hue='Gender')
 
plt.show()


# In[95]:


sns.barplot('Occupation','Purchase',hue='Gender',data=ds)


# In[ ]:


ds['Occupation'].unique()


# In[96]:


sns.barplot('Product_Category_1','Purchase',hue='Gender',data=ds)


# In[98]:


sns.barplot('Product_Category_2','Purchase',hue='Gender',data=ds)


# In[99]:


sns.barplot('Product_Category_3','Purchase',hue='Gender',data=ds)


# In[100]:


sns.heatmap(ds.corr(),annot=True)
plt.show()


# ### In this part, I will do more data cleaning because realized my model is not working... 

# In[141]:


df=dstrain.append(dstest)
df.head()


# In[142]:


df = ds.copy()


# In[145]:


from sklearn.preprocessing import LabelEncoder
lr = LabelEncoder()
df['Gender'] = lr.fit_transform(df['Gender'])
df['Age'] = lr.fit_transform(df['Age'])


# In[147]:


df.head()


# In[150]:


df['Product_Category_2'] =df['Product_Category_2'].fillna(0).astype('int64')
df['Product_Category_3'] =df['Product_Category_3'].fillna(0).astype('int64')
df['Purchase']= df['Purchase'].fillna(0).astype('int64')
df.isnull().sum()


# In[151]:


X = df.drop("Purchase",axis=1)
y=df['Purchase']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)


# ## Linear Regression

# In[160]:


model = LinearRegression()


# In[162]:


# Fit the model to the data
model.fit(df.drop("Purchase", axis=1), df["Purchase"])


# In[163]:


# Make predictions on the test set
predictions = model.predict(df.drop("Purchase", axis=1))


# In[164]:


# Evaluate the model
print(model.score(df.drop("Purchase", axis=1), df["Purchase"]))


# ### There is a 4.2876% chance of customer making a purchase.

# ## Random Forest

# In[166]:


from sklearn.ensemble import RandomForestRegressor


# In[167]:


# Create the random forest model
model = RandomForestRegressor(n_estimators=100, max_depth=5)

# Fit the model to the data
model.fit(df.drop("Purchase", axis=1), df["Purchase"])

# Make predictions on the test set
predictions = model.predict(df.drop("Purchase", axis=1))

# Evaluate the model
print(model.score(df.drop("Purchase", axis=1), df["Purchase"]))


# ### the R-squared value of 0.1987477346403813 means that the random forest model fits the data reasonably well. However, there is still some room for improvement. 
