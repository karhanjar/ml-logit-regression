#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


titanic_train = pd.read_csv('train.csv')


# In[4]:


titanic_train.head()


# In[5]:


titanic_train.isnull()


# In[6]:


sns.heatmap(titanic_train.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[7]:


sns.set_style('whitegrid')


# In[8]:


sns.countplot(x='Survived', data = titanic_train)


# In[9]:


sns.countplot(x='Survived', hue = 'Sex', data = titanic_train)


# In[10]:


sns.countplot(x='Survived', hue = 'Pclass', data = titanic_train)


# In[11]:


sns.distplot(titanic_train['Age'].dropna())


# In[12]:


sns.distplot(titanic_train['Age'].dropna(), kde=False, bins=30)


# In[13]:


titanic_train.info()


# In[16]:


sns.countplot(data=titanic_train, x='SibSp')


# In[17]:


titanic_train['Fare'].hist()


# In[20]:


sns.distplot(titanic_train['Fare'], kde=False , bins = 10)


# In[22]:


#import cufflinks as cf


# In[26]:


plt.figure(figsize=(7,10))
sns.boxplot(x='Pclass',y='Age',data=titanic_train)


# In[27]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        
        else:
            return 24
        
    else :
        return Age


# In[29]:


titanic_train['Age'] = titanic_train[['Age','Pclass']].apply(impute_age, axis = 1)


# In[31]:


sum(titanic_train['Age'].isnull())


# In[32]:


sns.heatmap(titanic_train.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[33]:


titanic_train.drop('Cabin',axis = 1 , inplace=True)


# In[34]:


sns.heatmap(titanic_train.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[36]:


titanic_train.dropna(inplace=True)


# In[37]:


sns.heatmap(titanic_train.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[39]:


titanic_train.head(5)


# In[45]:


sex = pd.get_dummies(titanic_train['Sex'],drop_first=True)


# In[46]:


embark = pd.get_dummies(titanic_train['Embarked'],drop_first=True)


# In[47]:


titanic_train=pd.concat([titanic_train, sex, embark], axis=1)


# In[48]:


titanic_train.head(5)


# In[49]:


titanic_train.drop(['Embarked','Sex','Name','Ticket'], axis=1, inplace=True)


# In[50]:


titanic_train_df = titanic_train.copy(deep=True)


# In[51]:


titanic_train_df.drop(['PassengerId'],axis=1, inplace=True)


# In[53]:


titanic_train_df.head(5)


# In[55]:


X = titanic_train_df.drop(['Survived'],axis=1)
y = titanic_train_df['Survived']


# In[57]:


from sklearn.model_selection import train_test_split


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 101)


# In[62]:


from sklearn.linear_model import LogisticRegression


# In[63]:


logmodel = LogisticRegression()


# In[64]:


logmodel.fit(X_train,y_train)


# In[65]:


predictions = logmodel.predict(X_test)


# In[66]:


from sklearn.metrics import classification_report


# In[70]:


print(classification_report(y_test,predictions))


# In[71]:


from sklearn.metrics import confusion_matrix


# In[73]:


print(confusion_matrix(y_test, predictions))

Using Pclass as categorical
# In[76]:


pclass = pd.get_dummies(titanic_train_df['Pclass'],drop_first=True)


# In[77]:


titanic_train_df = pd.concat([titanic_train_df,pclass],axis=1)


# In[78]:


titanic_train_df.head()


# In[79]:


titanic_train_df.drop('Pclass', axis=1, inplace=True)


# In[80]:


X = titanic_train_df.drop(['Survived'],axis=1)
y = titanic_train_df['Survived']


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state = 101)


# In[82]:


logmodel = LogisticRegression()


# In[83]:


logmodel.fit(X_train,y_train)


# In[85]:


predictions = logmodel.predict(X_test)


# In[86]:


print(classification_report(y_test,predictions))


# In[87]:


print(confusion_matrix(y_test, predictions))

Using Pclass as categorical has improved the prediction accuracy marginally-- Now using actual test data provided by them --
# In[129]:


titanic_test = pd.read_csv('test.csv')


# In[89]:


titanic_test.info()


# In[90]:


sns.heatmap(titanic_test.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[91]:


plt.figure(figsize=(7,10))
sns.boxplot(x='Pclass',y='Age',data=titanic_test)


# In[92]:


def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 42
        
        elif Pclass == 2:
            return 26
        
        else:
            return 24
        
    else :
        return Age


# In[130]:


titanic_test['Age'] = titanic_test[['Age','Pclass']].apply(impute_age,axis=1)


# In[94]:


sns.heatmap(titanic_test.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[131]:


titanic_test.drop(['Cabin'],axis =1, inplace=True)


# In[98]:


sns.heatmap(titanic_test.isnull(), yticklabels=False, cbar=False,cmap ='viridis')


# In[99]:


titanic_test.head(5)


# In[132]:


sex = pd.get_dummies(titanic_test['Sex'],drop_first=True)
embark = pd.get_dummies(titanic_test['Embarked'],drop_first=True)
pclass=pd.get_dummies(titanic_test['Pclass'],drop_first=True)


# In[133]:


titanic_test = pd.concat([titanic_test,sex,embark,pclass],axis=1)


# In[134]:


titanic_test.drop(['Name','Ticket','PassengerId','Sex','Embarked','Pclass'],axis =1 , inplace=True)


# In[135]:


X = titanic_train_df.drop(['Survived'],axis=1)
y = titanic_train_df['Survived']


# In[136]:


logmodel = LogisticRegression()


# In[137]:


logmodel.fit(X,y)


# In[138]:


#predictions = logmodel.predict(titanic_test)
titanic_test.dropna(inplace=True)


# In[139]:


predictions = logmodel.predict(titanic_test)


# In[140]:


titanic_test['Survived'] = predictions


# In[141]:


titanic_test.head(5)

