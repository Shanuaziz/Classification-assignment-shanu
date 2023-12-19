#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[7]:


data=sns.load_dataset('iris')


# In[8]:


data.head()


# In[9]:


data.columns


# In[10]:


X=data.iloc[:,:-1]


# In[11]:


y=data['species']


# In[12]:


from sklearn.model_selection import train_test_split


# In[13]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[14]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[15]:


svc_classifier=SVC()
svc_classifier.fit(X_train,y_train)


# In[16]:


knn_classifier=KNeighborsClassifier()
knn_classifier.fit(X_train,y_train)


# In[17]:


svc_predictions= svc_classifier.predict(X_test)
knn_predictions=knn_classifier.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score


# In[19]:


svc_accuracy=accuracy_score(y_test, svc_predictions)
knn_accuracy=accuracy_score(y_test, knn_predictions)


# In[20]:


print("Support Vector Classifier Accuracy:", svc_accuracy)
print("K Nearest Neighbors Classifier Accuracy:", knn_accuracy)


# In[21]:


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[22]:


print(confusion_matrix(svc_predictions,y_test))
print(classification_report(knn_predictions,y_test))


# In[1]:


# Random Forest method


# In[5]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[23]:


data=sns.load_dataset('iris')


# In[24]:


data.head()


# In[26]:


data['species'].value_counts()


# In[27]:


data.isnull().sum()


# In[28]:


X = data.drop(['species'], axis=1)


# In[29]:


y=data['species']


# In[30]:


from sklearn.model_selection import train_test_split


# In[31]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# In[32]:


X_train.shape, X_test.shape


# In[33]:


X_train.head(5)


# In[35]:


get_ipython().system('pip install category_encoders')


# In[36]:


import category_encoders as ce


# In[37]:


encoder = ce.OrdinalEncoder(cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
X_train = encoder.fit_transform(X_train)
X_test = encoder.transform(X_test)


# In[38]:


X_train.head()


# In[39]:


from sklearn.ensemble import RandomForestClassifier


# In[40]:


rfc = RandomForestClassifier(random_state=0)


# In[41]:


rfc.fit(X_train, y_train)


# In[42]:


y_pred = rfc.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score


# In[44]:


print('Model accuracy score with 10 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# In[45]:


# Decsion Tree


# In[46]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[47]:


data=sns.load_dataset('iris')


# In[48]:


data.head()


# In[50]:


data.shape


# In[51]:


data.info()


# In[52]:


data.isnull().sum()


# In[53]:


feature_cols=['petal_length','petal_width','sepal_length','sepal_width']
X = data[feature_cols] 
y = data.species


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[78]:


from sklearn.tree import DecisionTreeClassifier


# In[79]:


clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)


# In[80]:


clf = clf.fit(X_train,y_train)


# In[81]:


y_pred = clf.predict(X_test)


# In[82]:


from sklearn import metrics


# In[83]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[84]:


from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


# In[85]:


labels=['petal_length','petal_width','sepal_length','sepal_width']


# In[86]:


result=confusion_matrix(y_test,y_pred)
result


# In[2]:


#Naive Bayes


# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns


# In[4]:


data=sns.load_dataset('iris')


# In[5]:


data.head()


# In[7]:


feature_cols=['petal_length','petal_width','sepal_length','sepal_width']
X = data[feature_cols] 
y = data.species


# In[8]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[11]:


from sklearn.naive_bayes import GaussianNB


# In[12]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[13]:


y_pred = gnb.predict(X_test)


# In[14]:


from sklearn import metrics


# In[16]:


print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred))


# In[ ]:




