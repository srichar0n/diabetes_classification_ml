#!/usr/bin/env python
# coding: utf-8

# In[1]:


#capstone project on diabetes prediction using all the classification algorithms


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import warnings
warnings.simplefilter("ignore")


# In[4]:


df = pd.read_csv("C:\\Users\\Sricharan Reddy\\Downloads\\diabetes (1).csv")


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.corr()


# In[9]:


df.isnull().sum()


# In[10]:


#there are no null values in this data set


# In[11]:


df.duplicated().sum()


# In[12]:


#there is no duplicate data in this data set


# In[13]:


#in this data all the values are numerical there is no need of encoding 


# In[14]:


#there is no wrong data too


# In[15]:


x = df.drop(columns=['Outcome'])


# In[16]:


y = df['Outcome']


# In[17]:


from sklearn.model_selection import train_test_split


# In[66]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[67]:


#now lets do modelling 


# # Logistic Regression

# In[69]:


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


# # KNN Classification

# In[70]:


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


# # Decision trees

# In[71]:


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


# In[26]:


#here we can observe clearly there is a overfitting problem


# # Random Forests

# In[72]:


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


# In[28]:


#the testing accuracy and cross val score is improved compared to decision trees but still there is a overfitting problem


# # ADAboost

# In[73]:


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


# # Gradient boost

# In[74]:


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


# # XGBOOST

# In[35]:


get_ipython().system('pip install xgboost')


# In[76]:


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


# In[77]:


from sklearn.model_selection import GridSearchCV


# In[78]:


estimator = XGBClassifier()


# In[79]:


param_grid = {'n_estimators':[10,20,30,40,50],'max_depth':[1,2,3,4,5,6,7,8],'gamma':[0,0.15,0.30,0.5,0.7,1]}


# In[80]:


grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')


# In[81]:


grid.fit(x_train,y_train)


# In[82]:


grid.best_params_


# rebuilding the xgboost model with the best parameters

# In[83]:


from xgboost import XGBClassifier

xgmodel = XGBClassifier(gamma=0,max_depth= 2, n_estimators=10)
xgmodel.fit(x_train,y_train)

y_pred_train = xgmodel.predict(x_train)
y_pred_test = xgmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(xgmodel,x,y,cv=5).mean())


# In[53]:


#no improvement in accuracy


# In[54]:


#lets try doing hyper parameter tuning for the gradient boost algorithm


# In[84]:


estimator = GradientBoostingClassifier()


# In[85]:


param_grid = {'n_estimators':[10,20,30,40,50,60],'learning_rate':[0,0.1,0.3,0.5,0.7,1]}


# In[86]:


grid = GridSearchCV(estimator,param_grid,cv=5,scoring='accuracy')


# In[87]:


grid.fit(x_train,y_train)


# In[88]:


grid.best_params_


# In[61]:


#rebuilding the gradient boosting algorithm with the best parameters


# In[89]:


from sklearn.ensemble import GradientBoostingClassifier

gbmodel = GradientBoostingClassifier(learning_rate=0.1,n_estimators=10)
gbmodel.fit(x_train,y_train)

y_pred_train = gbmodel.predict(x_train)
y_pred_test = gbmodel.predict(x_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,y_pred_train))
print(accuracy_score(y_test,y_pred_test))

from sklearn.model_selection import cross_val_score
print(cross_val_score(gbmodel,x,y,cv=5).mean())


# In[63]:


#the accuracy has not been improved 


# In[ ]:


#till now we got the best accuracy with the gradient boosting algorithm with accuracy 
'''0.7719869706840391
0.8246753246753247
0.7695696460402341'''


# In[91]:


df.head()


# In[97]:


sns.boxplot(x='Outcome',y='Pregnancies',data=df)


# In[ ]:




