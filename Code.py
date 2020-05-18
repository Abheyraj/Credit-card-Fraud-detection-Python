#!/usr/bin/env python
# coding: utf-8

# In[4]:


#importing libraries requreid for fraud detection
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report


# In[194]:


#loading data set
#data = pd.read_csv(r'C:\Users\User\Downloads\creditcardfraud (1)\creditcard.csv')
#print(data.columns)


# In[3]:


#print(data.isnull().sum())


# In[4]:


#data.info()


# In[5]:


#dataset is shown as 
#data 


# In[6]:


#data size with rows and columns
#data.shape


# In[4]:


#data_fraud=data.loc[data['Class'] == 1]


# In[5]:


#data_fraud


# In[10]:


#data_notfraud=data.loc[data['Class'] == 0]


# In[11]:


#data_notfraud.head(10)


# In[12]:


#newdata_notfraud=data_notfraud.head(2500)


# In[13]:


#df_row = pd.concat([df1, df2])


# In[15]:


#new_data = pd.concat([newdata_notfraud,data_fraud])


# In[14]:


#new_data.head(20)


# In[17]:


#new_data.to_csv (r'C:\Users\User\Downloads\creditcardfraud (1)\newcreditcard.csv', index = False, header=True)


# In[5]:


new_data=pd.read_csv('newcreditcard.csv')


# In[6]:


new_data


# In[7]:


print(new_data.columns)


# In[8]:


print(new_data.isnull().sum())


# In[9]:


sns.countplot(x='Class', data=new_data)


# In[10]:


new_data.head(10)


# In[11]:


new_data.corr()


# In[12]:


# identifyig the data containing fraud and legal data
fraud = new_data[new_data['Class']== 1]
No_fraud = new_data[new_data['Class']== 0]


# In[13]:


# identifying statistics from the data  
print(new_data.describe())


# In[14]:


# data is visualized using histograms
new_data.hist(figsize= (30,30))
plt.show()


# In[15]:


print(fraud)
print(No_fraud)


# In[16]:


print('Number of fraud cases:',len(fraud))
print('Number of legal cases:',len(No_fraud))
outlier_frac = len(fraud)/float(len(No_fraud))
print(outlier_frac)


# In[17]:


n=len(new_data.columns)


# In[18]:


#number of variables for heatmap
cols = new_data.corr().nlargest(n, 'Class')['Class'].index
dj = new_data[cols].corr()
plt.figure(figsize=(19,10))
print(sns.heatmap(dj, annot=True, cmap = 'viridis'))


# In[19]:


#feature selection

time=new_data["Time"].values


new_data.drop(columns="Time",axis=1,inplace=True)


print(new_data.head(3))


# In[20]:


cols = new_data.columns.tolist()
cols = [c for c in cols if c not in ['Class']]
target = 'Class'
X = new_data[cols]
Y = new_data[target]
print(X.shape)
print(Y.shape)


# In[21]:


X


# In[22]:


from sklearn.model_selection import train_test_split
training_data, testing_data = train_test_split(new_data, test_size = 0.2)


# In[23]:


training_data


# In[24]:


testing_data


# In[25]:


x = training_data.iloc[:,:-1]


# In[26]:


x


# In[27]:


y=training_data.iloc[:,-1]


# In[28]:


y


# In[29]:


a=testing_data.iloc[:,:-1]


# In[30]:


a


# In[31]:


b=testing_data.iloc[:,-1]


# In[32]:


b


# In[33]:


training_input_data=x.values.tolist()


# In[34]:


training_output_data=y.values.tolist()


# In[35]:


testing_input_data=a.values.tolist()


# In[36]:


testing_output_data=b.values.tolist()


# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lgsregression = LogisticRegression(random_state=0,solver='lbfgs').fit(training_input_data,training_output_data)


# In[38]:


lgsregression_result = lgsregression.predict(testing_input_data)


# In[39]:


tn_lgs, fp_lgs, fn_lgs, tp_lgs = confusion_matrix(testing_output_data,lgsregression_result).ravel()


# In[44]:


accuracy_lgr=((tp_lgs + tn_lgs)/float((tn_lgs + tp_lgs + fn_lgs + fp_lgs)))


# In[45]:


accuracy_lgr


# In[47]:


recall_lgr=(tp_lgs)/float((tp_lgs+fn_lgs))


# In[48]:


recall_lgr


# In[49]:


precision_lgr=((tp_lgs)/float((tp_lgs+fp_lgs)))


# In[50]:


precision_lgr


# In[51]:


result = pd.DataFrame({'model': ["Lgr"],
                           'accuracy': [accuracy_lgr],'Recall': [recall_lgr],'precision': [precision_lgr]})


# In[52]:


result


# In[53]:


from sklearn import svm
svmclf = svm.SVC(gamma = 'scale')
   
svmclf.fit(training_input_data,training_output_data)
svmresult_list = svmclf.predict(testing_input_data)
tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(testing_output_data,svmresult_list).ravel()


# In[54]:


recall_svm=(tp_svm)/float((tp_svm+fn_svm))
precision_svm=((tp_svm)/float((tp_svm+fp_svm)))
accuracy_svm=((tp_svm + tn_svm)/float((tn_svm + tp_svm + fn_svm + fp_svm)))


# In[72]:


precision_svm


# In[73]:


recall_svm


# In[74]:


result = pd.DataFrame({'models': ["Lgr","SVM"],
                           'accuracy': [accuracy_lgr,accuracy_svm],'Recall': [recall_lgr,recall_svm],'precision': [precision_lgr,precision_svm]})


# In[71]:


result

