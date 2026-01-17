#!/usr/bin/env python
# coding: utf-8

# In[206]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[207]:


# Generate sample data

df = pd.read_csv('data/classification_dataset.csv')




# In[208]:


# transpose the dataframe to see the whole data at once
# df.head().T


# In[209]:


# data cleaning
df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_cols = list(df.dtypes[df.dtypes == 'object'].index)

for col in categorical_cols:
    df[col] = df[col].str.lower().str.replace(' ', '_')
    



# In[210]:


tc = pd.to_numeric(df.totalcharges, errors='coerce')


# In[211]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')


# In[212]:


df.totalcharges = df.totalcharges.fillna(0)

df.churn


# In[213]:


# replace target variable with binary values
df.churn  = (df.churn == 'yes').astype(int) 



# In[214]:


# SETTING UP THE VALIDATION FRAMEWORK

# perform the train-test split with scikit-learn
# what is skit-learn?   
from sklearn.model_selection import train_test_split




# In[215]:


# train_test_split splits data into train and test sets

df_full_train, df_test =  train_test_split(df, test_size=0.2, random_state=1)

len(df_full_train), len(df_test)


# In[216]:


# get validation dataset

df_train, df_val =  train_test_split(df_full_train, test_size=0.25, random_state=1)

len(df_train), len(df_val), len(df_test)


# In[217]:


# make indices not to be shuffles

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[218]:


# get target variable
y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


# In[219]:


# delete target variable from dataframes

del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[220]:


# EDA

# checking missing values
# look at the target variable (churn) distribution
# look at numerical and categorical variables






# In[221]:


df_full_train = df_full_train.reset_index(drop=True)



# In[222]:


# check missing values

df_full_train.isnull().sum()


# In[223]:


df_full_train.churn.value_counts(normalize=True)

# churn rate : the rate at which customers leave a service


# In[224]:


global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)


# In[225]:


df_full_train.dtypes


# In[226]:


# numerical columns

numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[227]:


df_full_train.columns


# In[228]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
        'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[229]:


df_full_train[categorical].nunique()


# In[230]:


df_full_train[numerical].nunique()


# In[231]:


# FEATURE IMPORTANCE: CHURN RATE AND RISK RATIO

# CHURN RATE
# RISK  RATIO
# MUTUTAL INFORMATION





# In[232]:


# CHURN RATE

churn_female =  df_full_train[df_full_train.gender == 'female'].churn.mean()

churn_female


# In[233]:


churn_male =  df_full_train[df_full_train.gender == 'male'].churn.mean()

churn_male


# In[234]:


churn_partner =  df_full_train[df_full_train.partner == 'yes'].churn.mean()

churn_partner


# In[235]:


churn_no_partner =  df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner


# In[236]:


global_churn = df_full_train.churn.mean()

global_churn


# In[237]:


churn_female_difference = global_churn - churn_female
churn_male_difference = global_churn - churn_male


# partner

churn_partner_difference = global_churn - churn_partner

churn_no_partner_difference = global_churn - churn_no_partner


print(churn_female_difference)
print(churn_male_difference)


# partner
print(churn_partner_difference)
print(churn_no_partner_difference)


# In[238]:


# RISK RATIO

churn_no_partner / global_churn


# In[239]:


churn_partner / global_churn


# In[240]:


from IPython.display import display


# In[241]:


# we cannot individually calculate risk ratio for all categorical variables
# so we can leverage on sql like groupby functionality of pandas

# SELECT gender, AVG(churn), AVG(churn), AVG(churn) - global_churn as diff, AVG(churn) / global_churn as risk_ratio FROM data GROUP BY gender

df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group['diff'] = df_group['mean'] - global_churn
df_group['risk_ratio'] = df_group['mean'] / global_churn
df_group

for col in categorical:
    df_group = df_full_train.groupby(col).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk_ratio'] = df_group['mean'] / global_churn
    print(f'\nColumn: {col}')
    display(df_group)




# In[242]:


# Feature importance : Mutual Information

# concept from information theory that tells us how much information about one variable can be obtained by observing another variable

from sklearn.metrics import mutual_info_score

# mutual_info_score helps us to compute mutual information between two discrete variables
# comparing churn dataset with contract type

mutual_info_score(df_full_train.churn, df_full_train.contract)


# In[243]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# In[244]:


df_full_train[categorical].apply(mutual_info_churn_score).sort_values(ascending=False)


# In[245]:


# Feature correlation 
# Mutual information : a way to measure importance of categorical features
# correlation : a way to measure importance of numerical features

# what is correlation?

# A way to measure importance of numerical features is to calculate their correlation with the target variable



# In[246]:


df_full_train[numerical].corrwith(df_full_train.churn)

# df_full_train[numerical].corrwith(df_full_train.churn).abs()


# In[247]:


# one-hot encoding
# use scikit-learn's to encode categorical variables


from sklearn.feature_extraction import DictVectorizer


# In[248]:


train_dicts = (df_train[categorical + numerical].to_dict(orient='records'))


# In[249]:


dv = DictVectorizer(sparse=False)


# In[250]:


# get feature names: dv.get_feature_names_out()
x_train = dv.fit_transform(train_dicts)




# In[251]:


val_dicts = df_val[categorical + numerical].to_dict(orient='records')
x_val = dv.transform(val_dicts)


# In[252]:


# test_dicts = df_test[categorical + numerical].to_dict(orient='records')
# x_test = dv.transform(test_dicts)


# In[253]:


# Logistic regression model

from sklearn.linear_model import LogisticRegression


# In[254]:


# Train the model

model = LogisticRegression()

model.fit(x_train, y_train)


# In[255]:


from sklearn.metrics import accuracy_score


# In[256]:


y_pred = model.predict_proba(x_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()  # accuracy


# In[257]:


accuracy_score(y_val, y_pred >= 0.5)


# In[258]:


# Accuracy
# correct predictions / total predictions

(y_val == churn_decision).sum() / len(y_val)


# In[259]:


# evaluate prediction for different thresholds i.e from 0 to 21 
thresholds = np.linspace(0, 1, 21)
scores = []
for t in thresholds:
    score = accuracy_score(y_val, (y_pred >= t).astype(int))
    print('%.2f %.3f' % (t, score))
    scores.append(score)


# In[260]:


plt.plot(thresholds, scores)


# In[261]:


from collections import Counter

Counter(y_pred >= 1.0)


# In[262]:


#CONFUSION TABLE
# A way of looking at error and prediction scores
actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


# In[263]:


t = 0.5
predicted_positive = (y_pred >= t)
predictive_negative = (y_pred < t)


# In[264]:


tp = (predicted_positive & actual_positive).sum()
tn = (predictive_negative & actual_negative).sum()


# In[265]:


fp = (predicted_positive & actual_negative).sum()
fn = (predictive_negative & actual_positive).sum()
fp, fn


# In[266]:


confusion_matrix = np.array([[tn, fp], [fn, tp]])

(confusion_matrix  / confusion_matrix.sum()).round(2)


# In[267]:


(tp + tn) / (tp + tn + fp + fn)  # accuracy


# In[268]:


# PRECISION

# TELL USE THE FRACTION OF POSITIVE PREDICTIONS THAT ARE CORRECT

p = tp / (tp + fp)


p


# In[269]:


# RECALL
# TELL US THE FRACTION OF ACTUAL POSITIVES THAT ARE CORRECTLY IDENTIFIED BY THE MODEL 

r = tp / (tp + fn)

r


# In[270]:


# ROC CURVE AND AUC

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[271]:


plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('False Positive Rate as FPR')
plt.ylabel('True Positive Rate as TPR')


# In[272]:


# calculate AUC

from sklearn.metrics import auc


# In[273]:


auc_value = auc(fpr, tpr)

auc_value


# In[274]:


from sklearn.metrics import roc_auc_score   
roc_auc_score(y_val, y_pred)


# In[275]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[276]:


import random


# In[277]:


n = 100000
success = 0
for i in range(n):
    pos_index = random.randint(0, len(pos) - 1)
    neg_index = random.randint(0, len(neg) - 1)
    
    if pos[pos_index] > neg[neg_index]:
        success = success + 1

success / n


# In[278]:


n = 500000

np.random.seed(1)
pos_index = np.random.randint(0, len(pos), size=n)
neg_index = np.random.randint(0, len(neg), size=n)


# In[279]:


(pos[pos_index] > neg[neg_index]).mean()


# In[280]:


# CROSS VALIDATION: A WAY TO ESTIMATE MODEL PERFORMANCE ON DIFFERENT DATASETS
# K-FOLD CROSS VALIDATION
# Evaluating the same model on different subset  of data
# Getting the average prediction and the spread within the predictions


def train(df_train, y_train, C=1.0):
    
    dicts = (df_train[categorical + numerical].to_dict(orient='records'))
    
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(x_train, y_train)
    return dv, model




# In[281]:


dv, model = train(df_train,y_train, C=0.001)


# In[282]:


def predict(df, dv, model):
    dicts = (df[categorical + numerical].to_dict(orient='records'))
    x = dv.transform(dicts)
    y_pred = model.predict_proba(x)[:, 1]
    return y_pred


# In[283]:


y_pred = predict(df_val, dv, model)


# In[284]:


from sklearn.model_selection import KFold


# In[285]:


kfold = KFold(n_splits=10, shuffle=True, random_state=1)


# In[286]:


from tqdm import tqdm


# In[287]:


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []
    for train_idx, val_idx  in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
    
        y_train = df_full_train.churn.iloc[train_idx].values
        y_val = df_full_train.churn.iloc[val_idx].values
    
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
    
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[288]:


print('%.3f +- %.3f' % (np.mean(scores), np.std(scores)))


# In[289]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
    
auc = roc_auc_score(y_test, y_pred)

auc


# In[290]:


# when to use cross validation?
# when u have small dataset


# In[291]:


# Save the model with Pickle, what is pickle?

import pickle


# In[292]:


output_file = f'model_C={C}.bin'
output_file


# In[293]:


#save our model and save it to a file

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    # do stuff
    
# do other stuff


# In[294]:


#Load the model from the file

with open(output_file, 'rb') as f_in:
    dv,  model =    pickle.load(f_in)


# In[303]:


customer = {
    'gender' : 'female',
    'seniorcitizen' : 0,
    'partner' : 'yes',
    'dependents' : 'no',
    'phoneservice' : 'no',
    'multiplelines' : 'no_phone_service',
    'internetservice' : 'dsl',
    'onlinesecurity' : 'no',
    'onlinebackup' : 'yes',
    'deviceprotection' : 'no',
    'techsupport' : 'no',
    'streamingtv' : 'no',
    'streamingmovies' : 'no',
    'contract' : 'month-to-month',
    'paperlessbilling' : 'yes',
    'paymentmethod' : 'electronic_check',
    'tenure' : 1,
    'monthlycharges' : 29.85,
    'totalcharges' : 29.85
}


# In[304]:


X  = dv.transform([customer])


# In[305]:


model.predict_proba(X)[0, 1]


# In[ ]:





# In[ ]:





# In[ ]:




