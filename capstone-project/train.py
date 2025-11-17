#!/usr/bin/env python
# coding: utf-8

# In[201]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from IPython.display import display
from sklearn.metrics import mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score  
from sklearn.metrics import auc 
from sklearn.model_selection import KFold
from tqdm import tqdm
import pickle




# In[202]:


df1 = pd.read_csv('data/ecommerce_customer_behavior_dataset.csv')
df2 = pd.read_csv('data/ecommerce_customer_behavior_dataset_v2.csv')

print(df1.columns.equals(df2.columns))
duplicates = set(df1['Customer_ID']).intersection(set(df2['Customer_ID']))
print(len(duplicates))


# In[203]:


# combine datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Remove duplicate orders if any (very safe and recommended)
combined_df = combined_df.drop_duplicates(subset='Order_ID')

combined_df['Customer_ID'].nunique()  # number of unique customers

combined_df.isnull().sum()  # check for missing values
combined_df.dtypes

combined_df.head(100)



# In[204]:


def analyze_duplicates(df):
    """
    Comprehensive duplicate analysis
    """
    print("="*60)
    print("DUPLICATE ANALYSIS")
    print("="*60)
    
    # 1. Exact duplicates (all columns)
    exact_dups = df.duplicated().sum()
    print(f"\n1. EXACT DUPLICATES (all columns): {exact_dups}")
    
    # 2. Order ID duplicates
    order_dups = df['Order_ID'].duplicated().sum()
    print(f"\n2. DUPLICATE ORDER_IDs: {order_dups}")
    if order_dups > 0:
        print("   ⚠️ WARNING: Order_ID should be unique!")
        print(f"   Repeated Order_IDs: {df[df['Order_ID'].duplicated(keep=False)]['Order_ID'].unique()[:5]}")
    
    # 3. Customer duplicates on same date
    same_day_dups = df.duplicated(subset=['Customer_ID', 'Date'], keep=False).sum()
    print(f"\n3. SAME CUSTOMER + SAME DAY: {same_day_dups}")
    print(f"   (Normal if customer made multiple purchases same day)")
    
    # 4. Suspicious duplicates (same customer, date, amount)
    suspicious = df.duplicated(subset=['Customer_ID', 'Date', 'Total_Amount'], keep=False).sum()
    print(f"\n4. SUSPICIOUS DUPLICATES (Customer + Date + Amount): {suspicious}")
    if suspicious > 0:
        print("   ⚠️ These might be data entry errors!")
    
    # 5. Potential duplicates (same customer, date, product, amount)
    potential = df.duplicated(
        subset=['Customer_ID', 'Date', 'Product_Category', 'Total_Amount'], 
        keep=False
    ).sum()
    print(f"\n5. POTENTIAL DUPLICATES (Customer + Date + Product + Amount): {potential}")
    
    # 6. Check each key column
    print("\n6. DUPLICATE COUNTS BY KEY COLUMNS:")
    for col in ['Order_ID', 'Customer_ID', 'Date']:
        unique_count = df[col].nunique()
        total_count = len(df)
        dup_count = total_count - unique_count
        print(f"   {col}: {unique_count} unique / {total_count} total ({dup_count} duplicates)")
    
    return {
        'exact_duplicates': exact_dups,
        'order_id_duplicates': order_dups,
        'suspicious_duplicates': suspicious
    }

# Run the analysis
dup_stats = analyze_duplicates(combined_df)


# In[205]:


# # try different churn definitions

# # Ensure Date column is datetime type
# combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# # Get the last date in your dataset
# last_date = combined_df['Date'].max()

# # Compute last purchase per customer
# customer_last_purchase = combined_df.groupby('Customer_ID')['Date'].max().reset_index()
# customer_last_purchase.columns = ['Customer_ID', 'Last_Purchase_Date']

# # Calculate days since last purchase
# customer_last_purchase['days_since_last_purchase'] = (last_date - customer_last_purchase['Last_Purchase_Date']).dt.days

# # Test multiple thresholds
# thresholds = [60, 90, 120, 180]

# for threshold in thresholds:
#     customer_last_purchase[f'Churn_{threshold}d'] = (
#         customer_last_purchase['days_since_last_purchase'] >= threshold
#     ).astype(int)
    
#     churn_rate = customer_last_purchase[f'Churn_{threshold}d'].mean() * 100
#     print(f"{threshold} days: {churn_rate:.1f}% churn rate")

# Implementing a single churn definition: 90 days
# A customer who hasn’t purchased in the last 90 days is considered churned (1), otherwise active (0).

# Ensure Date column is datetime type
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Get the last date in your dataset
last_date = combined_df['Date'].max()

# Compute last purchase per customer
customer_last_purchase = combined_df.groupby('Customer_ID')['Date'].max().reset_index()
customer_last_purchase.columns = ['Customer_ID', 'Last_Purchase_Date']

# Calculate days since last purchase
customer_last_purchase['days_since_last_purchase'] = (last_date - customer_last_purchase['Last_Purchase_Date']).dt.days

# Create churn column using 90-day threshold
churn_threshold = 90
customer_last_purchase['Churn_90d'] = (
    customer_last_purchase['days_since_last_purchase'] >= churn_threshold
).astype(int)

# print churn rate for 90 days
churn_rate = customer_last_purchase['Churn_90d'].mean() * 100
print(f"90 days: {churn_rate:.1f}% churn rate")

# Merge churn column back into transaction-level data
combined_df = combined_df.merge(
    customer_last_purchase[['Customer_ID', 'Churn_90d']],
    on='Customer_ID',
    how='left',
    suffixes=('', '_new')  # prevents _x/_y naming
)


# In[206]:


# EDA

# DATA OVERVIEW
print("Shape of dataset:", combined_df.shape)
print("\nData types:\n", combined_df.dtypes)
print("\nMissing values:\n", combined_df.isnull().sum())
print("\nSummary statistics:\n", combined_df.describe())


# In[207]:


# Bar plot for churn distribution
sns.countplot(data=combined_df, x='Churn_90d')
plt.title("Distribution of Churn (90 days)")
plt.xlabel("Churn_90d")
plt.ylabel("Number of Customers")
plt.show()


# In[208]:


# NUMERIC FEATURE ANALYSIS
numeric_cols = ['Age', 'Unit_Price', 'Quantity', 'Total_Amount', 
                'Session_Duration_Minutes', 'Pages_Viewed', 'Customer_Rating', 'Delivery_Time_Days']

# Histograms
combined_df[numeric_cols].hist(bins=20, figsize=(15,10))
plt.suptitle("Numeric Feature Distributions")
plt.show()

# Boxplots for numeric features vs churn
for col in numeric_cols:
    sns.boxplot(x='Churn_90d', y=col, data=combined_df)
    plt.title(f"{col} vs Churn")
    plt.show()

# Correlation heatmap
corr = combined_df[numeric_cols + ['Churn_90d']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation with Churn")
plt.show()


# In[209]:


# CATEGORICAL FEATURE ANALYSIS 
categorical_cols = ['Gender', 'City', 'Device_Type', 'Payment_Method', 'Product_Category', 'Is_Returning_Customer']

for col in categorical_cols:
    print(f"\nChurn rate by {col}:")
    print(combined_df.groupby(col)['Churn_90d'].mean().sort_values(ascending=False))
    
    sns.countplot(data=combined_df, x=col, hue='Churn_90d')
    plt.title(f"{col} vs Churn")
    plt.xticks(rotation=45)
    plt.show()


# In[210]:


df_full_train, df_test = train_test_split(combined_df, test_size=0.2, random_state=1)

# get validation dataset

df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1) 

# make indices not to be shuffles

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

len(df_train), len(df_val), len(df_test)


# In[211]:


# get target variable
y_train = df_train.Churn_90d.values
y_val = df_val.Churn_90d.values
y_test = df_test.Churn_90d.values

# delete target variable from dataframes

del df_train['Churn_90d']
del df_val['Churn_90d']
del df_test['Churn_90d']


# In[212]:


df_train.shape, df_val.shape, df_test.shape


# In[213]:


global_churn = df_full_train.Churn_90d.mean()


# In[214]:


# FEATURE IMPORTANCE: CHURN RATE AND RISK RATIO

# CHURN RATE
# RISK  RATIO
# MUTUTAL INFORMATION



# In[215]:


numeric_cols = ['Age', 'Unit_Price', 'Quantity', 'Total_Amount', 
                'Session_Duration_Minutes', 'Pages_Viewed', 'Customer_Rating', 'Delivery_Time_Days']

categorical_cols = ['Gender', 'City', 'Device_Type', 'Payment_Method', 'Product_Category', 'Is_Returning_Customer']


# In[216]:


for col in categorical_cols:
    df_group = df_full_train.groupby(col).Churn_90d.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn
    df_group['risk_ratio'] = df_group['mean'] / global_churn
    print(f'\nColumn: {col}')
    display(df_group)


# In[217]:


risk_ratios = []

for col in categorical_cols:
    group = df_full_train.groupby(col)['Churn_90d'].mean().reset_index()
    group['risk_ratio'] = group['Churn_90d'] / global_churn
    group['feature'] = col
    group.rename(columns={col: 'category'}, inplace=True)
    risk_ratios.append(group[['feature', 'category', 'risk_ratio']])

# Combine all
risk_df = pd.concat(risk_ratios, ignore_index=True)

# Sort within each feature for better visualization
risk_df.sort_values(['feature', 'risk_ratio'], ascending=[True, False], inplace=True)

# Plot
plt.figure(figsize=(12, 8))
sns.barplot(x='risk_ratio', y='category', hue='feature', data=risk_df, dodge=False)
plt.axvline(1, color='red', linestyle='--', label='Average Churn Risk')
plt.title('Churn Risk Ratios by Feature Categories')
plt.xlabel('Risk Ratio (vs Average Churn)')
plt.ylabel('Category')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# In[218]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.Churn_90d)


# In[219]:


df_full_train[categorical_cols].apply(mutual_info_churn_score).sort_values(ascending=False)


# In[220]:


# Train model:
dv = DictVectorizer(sparse=False)
train_dicts = (df_train[categorical_cols + numeric_cols].to_dict(orient='records'))
x_train = dv.fit_transform(train_dicts)
val_dicts = df_val[categorical_cols + numeric_cols].to_dict(orient='records')
x_val = dv.transform(val_dicts)

# Train the model

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42, class_weight='balanced')

model.fit(x_train, y_train)
y_pred = model.predict_proba(x_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()  # accuracy


# In[221]:


# evaluate prediction for different thresholds i.e from 0 to 21 
thresholds = np.linspace(0, 1, 21)
scores = []
for t in thresholds:
    score = accuracy_score(y_val, (y_pred >= t).astype(int))
    print('%.2f %.3f' % (t, score))
    scores.append(score)


# In[222]:


#CONFUSION TABLE
# A way of looking at error and prediction scores

t = 0.5

actual_positive = (y_val == 1)
actual_negative = (y_val == 0)


predicted_positive = (y_pred >= t)
predictive_negative = (y_pred < t)

tp = (predicted_positive & actual_positive).sum()
tn = (predictive_negative & actual_negative).sum()

fp = (predicted_positive & actual_negative).sum()
fn = (predictive_negative & actual_positive).sum()
tp, tn, fp, fn


# In[223]:


confusion_matrix = np.array([[tn, fp], [fn, tp]])

(confusion_matrix  / confusion_matrix.sum()).round(2) 


# In[224]:


thresholds = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5]
for t in thresholds:
    churn_decision = (y_pred >= t).astype(int)
    precision = precision_score(y_val, churn_decision)
    recall = recall_score(y_val, churn_decision)
    f1 = f1_score(y_val, churn_decision)
    print(f"Threshold {t:.2f}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")


# In[225]:


# ROC CURVE AND AUC
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('False Positive Rate as FPR')
plt.ylabel('True Positive Rate as TPR')


# In[226]:


auc_value = auc(fpr, tpr)
roc_auc_score(y_val, y_pred)


# In[227]:


# CROSS VALIDATION: A WAY TO ESTIMATE MODEL PERFORMANCE ON DIFFERENT DATASETS
# K-FOLD CROSS VALIDATION
# Evaluating the same model on different subset  of data
# Getting the average prediction and the spread within the predictions

def train(df_train, y_train, C=1.0):
    
    dicts = (df_train[categorical_cols + numeric_cols].to_dict(orient='records'))
    
    dv = DictVectorizer(sparse=False)
    x_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(x_train, y_train)
    return dv, model


# In[228]:


dv, model = train(df_train,y_train, C=0.001)


# In[229]:


def predict(df, dv, model):
    dicts = (df[categorical_cols + numeric_cols].to_dict(orient='records'))
    x = dv.transform(dicts)
    y_pred = model.predict_proba(x)[:, 1]
    return y_pred


# In[230]:


y_pred = predict(df_val, dv, model)


# In[231]:


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []
    for train_idx, val_idx  in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]
    
        y_train = df_full_train.Churn_90d.iloc[train_idx].values
        y_val = df_full_train.Churn_90d.iloc[val_idx].values
    
        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)
    
        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)
    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[232]:


dv, model = train(df_full_train, df_full_train.Churn_90d.values, C=1.0)
y_pred = predict(df_test, dv, model)
    
auc = roc_auc_score(y_test, y_pred)

auc


# In[233]:


output_file = f'model_C={C}.bin'
output_file


# In[234]:


#save  model and save it to a file

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)


# In[235]:


#Load the model from the file

with open(output_file, 'rb') as f_in:
    dv,  model =    pickle.load(f_in)


# In[238]:


customer = {
    'Age':32,
    'Gender': 'Male',
    'City': 'Istanbul',
    'Device_Type': 'Mobile',
    'Payment_Method': 'Credit Card',
    'Product_Category': 'Electronics',
    'Is_Returning_Customer': 'False',
    'Unit_Price': 804.06,
    'Quantity': 1,
    'Total_Amount': 574.78,
    'Session_Duration_Minutes': 8,
    'Pages_Viewed': 10,
    'Customer_Rating': 4,
    'Delivery_Time_Days':  1,
    'Discount_Amount': 229.28
}


# In[239]:


X = dv.transform([customer])

model.predict_proba(X)[0, 1]  # probability of churn


# In[ ]:




