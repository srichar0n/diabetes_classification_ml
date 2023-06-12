#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import warnings
warnings.simplefilter("ignore")


# In[3]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\diabetes (1).csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.duplicated().sum()


# In[8]:


df.corr()


# In[9]:


sns.boxplot(x='Outcome',y='Glucose',data=df)
plt.show()


# In[10]:


from feature_engine.outliers import Winsorizer


# In[11]:


win = Winsorizer(capping_method='iqr',tail='both',fold=1.5)


# In[12]:


df['Pregnancies'] = win.fit_transform(df[['Pregnancies']])


# In[13]:


sns.boxplot(x='Outcome',y='Pregnancies',data=df)
plt.show()


# In[14]:


df['Glucose'] = win.fit_transform(df[['Glucose']])


# In[15]:


sns.boxplot(x='Outcome',y='Glucose',data=df)
plt.show()


# In[16]:


sns.boxplot(x=df['Glucose'])


# In[17]:


sns.boxplot(x=df['BloodPressure'])


# In[18]:


df['BloodPressure'] = win.fit_transform(df[['BloodPressure']])


# In[19]:


sns.boxplot(x=df['BloodPressure'])


# In[20]:


df['SkinThickness'] = win.fit_transform(df[['SkinThickness']])


# In[21]:


sns.boxplot(x=df['SkinThickness'])


# In[22]:


sns.boxplot(x=df['DiabetesPedigreeFunction'])


# In[23]:


#df['DiabetesPedigreeFunction'] = win.fit_transform(df[['DiabetesPedigreeFunction']])


# In[24]:


sns.boxplot(x=df['DiabetesPedigreeFunction'])


# In[25]:


df['Insulin'] = win.fit_transform(df[['Insulin']])


# In[26]:


df['BMI'] = win.fit_transform(df[['BMI']])


# In[27]:


x = df.drop(columns=['Outcome'])


# In[28]:


y = df['Outcome']


# In[29]:


from sklearn.model_selection import train_test_split


# In[30]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# # Logistic Regression

# In[31]:


from sklearn.linear_model import LogisticRegression

lrmodel = LogisticRegression()
lrmodel.fit(x_train,y_train)

y_pred_train = lrmodel.predict(x_train)
y_pred_test = lrmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(lrmodel,x,y,cv=5).mean())


# # KNN classification

# In[32]:


from sklearn.neighbors import KNeighborsClassifier

knnmodel = KNeighborsClassifier()
knnmodel.fit(x_train,y_train)

y_pred_train = knnmodel.predict(x_train)
y_pred_test = knnmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(knnmodel,x,y,cv=5).mean())


# # Decision Trees

# In[33]:


from sklearn.tree import DecisionTreeClassifier

dtmodel = DecisionTreeClassifier()
dtmodel.fit(x_train,y_train)

y_pred_train = dtmodel.predict(x_train)
y_pred_test = dtmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(dtmodel,x,y,cv=5).mean())


# # Random Forests

# In[34]:


from sklearn.ensemble import RandomForestClassifier

rfmodel = RandomForestClassifier()
rfmodel.fit(x_train,y_train)

y_pred_train = rfmodel.predict(x_train)
y_pred_test = rfmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(rfmodel,x,y,cv=5).mean())


# # ADA boost

# In[35]:


from sklearn.ensemble import AdaBoostClassifier

admodel = AdaBoostClassifier()
admodel.fit(x_train,y_train)

y_pred_train = admodel.predict(x_train)
y_pred_test = admodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(admodel,x,y,cv=5).mean())


# # Gradient Boost

# In[36]:


from sklearn.ensemble import GradientBoostingClassifier

gbmodel = GradientBoostingClassifier()
gbmodel.fit(x_train,y_train)

y_pred_train = gbmodel.predict(x_train)
y_pred_test = gbmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(gbmodel,x,y,cv=5).mean())


# # XG boost

# In[37]:


from xgboost import XGBClassifier

xgmodel = XGBClassifier()
xgmodel.fit(x_train,y_train)

y_pred_train = xgmodel.predict(x_train)
y_pred_test = xgmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(xgmodel,x,y,cv=5).mean())


# # Till now the highest accuracy i got is with logistic regression that is 84% 

# In[ ]:




