
# coding: utf-8

# In[524]:


import pandas as pd
import numpy as np
import re


# # Reading data from csv

# In[525]:


data_Singapore = pd.read_csv("data/SingaporeTrain.csv")
data_NY = pd.read_csv("data/NYTrain.csv")
data_London = pd.read_csv("data/LondonTrain.csv")
data_Singapore.head()
# data_NY.head()
# data_London.head()


# # Combining all the data to a single dataframe

# In[526]:


data_Singapore['source'] = 'Singapore'
data_NY['source'] = 'NY'
data_London['source'] = 'London'
data = pd.concat([data_Singapore, data_NY, data_London],ignore_index=True)
data.shape


# In[527]:


data['source'].unique()


# In[528]:


data['gender'].unique()


# In[529]:


data['realAge'].unique()


# In[530]:


data['ageGroup'].unique()


# In[531]:


data['relationship'].unique()


# In[532]:


data['educationLevel'].unique()


# In[533]:


data['occupation'].unique()


# In[534]:


data['income'].unique()


# In[535]:


data['income'] = data['income'].fillna("no").apply(lambda x: -1 if x == "no" else len(str(x)))


# In[536]:


data['income'].unique()


# # Exctracting features from "workInfoForAgeGroupEstimation"

# In[537]:


row_id = 4
row = data['workInfoForAgeGroupEstimation'][row_id]
# row = 'PT. Duta Marga Lestarindo September 19 present Land Transport Authority Project Engineer � August 2011 to July 2013 T.Y. Lin International Civil Engineer � January 2010 to June 2010'
row


# In[538]:


def process_work_info(row):
    years_raw = re.compile(r'\b(?:19|20)\d{2}|\bpresent').findall(row)
    years =  [int(year) if year != 'present' else 2018 for year in years_raw]
    
    start_year, number_of_places, total_length, mean_length, min_length, max_length, working_now = None, None, None, None, None, None, None
    if row != None and len(years)<2:
        number_of_places = 1
    else:
        pairs = [years[i:i+2] for i in range(0, len(years)-1, 2)]
        lengths = [pair[1]-pair[0] for pair in pairs]
        number_of_places = len(pairs)
        max_length = max(lengths)
        min_length = min(lengths)
        total_length = sum(lengths)
        mean_length = total_length / float(len(lengths))
        start_year = min(years)
        working_now = None
        if len(years_raw)>0:
            working_now = float('present' in years_raw) 
            
    return start_year, number_of_places, total_length, mean_length, min_length, max_length, working_now


# In[539]:


process_work_info(row)


# In[540]:


data['workInfoForAgeGroupEstimation'] = data['workInfoForAgeGroupEstimation'].map(str).apply(process_work_info)
data = data.join(pd.DataFrame(data['workInfoForAgeGroupEstimation'].tolist(), index=data.index, columns=['w_start_year', 'w_number_of_places', 'w_total_length', 'w_mean_length', 'w_min_length', 'w_max_length', 'w_working_now']))
data.drop(columns=['workInfoForAgeGroupEstimation'], inplace=True)


# # Exctracting features from "educationInfoForAgeGroupEstimation"

# In[541]:


# possible_levels = ['college', 'school', 'undergraduate', 'graduate']

# possible_values_for_levels = {}
# "Secondary School", "Ngee", "Institute", "University", "High School"
# # CHIJ Secondary (Toa Payoh) - автономная католическая школа для девочек в Сингапуре. 
# Class of 2010


# In[542]:


row_id = 45
row = data['educationInfoForAgeGroupEstimation'][row_id]
# row = "Republic Polytechnic 2012 to 2015 � Singapore CHIJ Kellock Singapore CHIJ St. Theresa&#039;s Convent Class of 2011 � Singapore Republic Polytechnic Integrated Event Management � Singapore"
row


# In[543]:


def process_education_info(row):
    class_of_years = [int(i) for i in re.compile(r'(?<=of )\d{4}').findall(row)]
    range_years_starts = [int(i) for i in re.compile(r'\b(?:19|20)\d{2}(?= to )').findall(row)]
    range_years_ends = [int(i) if i != 'present' else 2018 for i in re.compile(r'(?<= to )(?:19|20)\d{2}|\bpresent').findall(row)]

    first_start_year = None
    last_start_year = None
    if len(class_of_years)>0 or len(range_years_starts)>0:
        first_start_year = min(class_of_years+range_years_starts)
        last_start_year = max(class_of_years+range_years_starts) 

    
    finish_year = None
    if len(range_years_ends)>0:
        finish_year = max(range_years_ends)
    # else
    #     if len(class_of_years)>0 | len(range_years_starts)>0:
    #     finish_year = max(class_of_years) + 4 #usually one program = 4year

    # study_now = None

    number_of_programs = None
    num = len(class_of_years)+len(range_years_starts)
    if num<1 and len(row)>0:
        number_of_programs = 1
    else:
        number_of_programs = num

    return first_start_year, last_start_year, finish_year, number_of_programs


# In[544]:


process_education_info(row)


# In[545]:


data['educationInfoForAgeGroupEstimation'] = data['educationInfoForAgeGroupEstimation'].map(str).apply(process_education_info)
data = data.join(pd.DataFrame(data['educationInfoForAgeGroupEstimation'].tolist(), index=data.index, columns=['e_first_start_year', 'e_last_start_year', 'e_finish_year', 'e_number_of_programs']))
data.drop(columns=['educationInfoForAgeGroupEstimation'], inplace=True)


# # Some preprocessing

# In[546]:


columns_to_hot = ['gender', 'relationship', 'educationLevel', 'occupation', 'source']
data = pd.get_dummies(data, columns=columns_to_hot)
data_with_labels = data[data['ageGroup'].notnull()].drop(columns=['row ID']).fillna(-1.0)

from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
data_with_labels['ageGroup'] = LE.fit_transform(data_with_labels['ageGroup'])


# In[547]:


data_with_labels.head()


# # MODELS
# Train-test split

# In[548]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_with_labels.drop(columns=['ageGroup']), data_with_labels.ageGroup, test_size=0.3, random_state=42)


# ## Classifiers

# In[549]:


from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, accuracy_score

rf_cls = RandomForestClassifier(n_estimators=100)
xgb_cls = XGBClassifier()


# ## Learning and scores
# RandomForestClassifier

# In[550]:


get_ipython().run_cell_magic('time', '', 'rf_cls.fit(X_train, y_train)\ny_pred_on_train = rf_cls.predict(X_train)\ny_pred_on_test = rf_cls.predict(X_test)\n\nprint("F1 score on train:", f1_score(y_train, y_pred_on_train, average=\'weighted\'))\nprint("F1 score on test:", f1_score(y_test, y_pred_on_test, average=\'weighted\'))\nprint("Accuracy on train:", accuracy_score(y_train, y_pred_on_train))\nprint("Accuracy on test:", accuracy_score(y_test, y_pred_on_test))')


# XGBClassifier

# In[551]:


get_ipython().run_cell_magic('time', '', 'xgb_cls.fit(X_train, y_train)\ny_pred_on_train = xgb_cls.predict(X_train)\ny_pred_on_test = xgb_cls.predict(X_test)\n\nprint("F1 score on train:", f1_score(y_train, y_pred_on_train, average=\'weighted\'))\nprint("F1 score on test:", f1_score(y_test, y_pred_on_test, average=\'weighted\'))\nprint("Accuracy on train:", accuracy_score(y_train, y_pred_on_train))\nprint("Accuracy on test:", accuracy_score(y_test, y_pred_on_test))')


# ## Confusion Matrix

# In[552]:


y_pred = rf_cls.predict(X_test)
reversefactor = dict(zip(range(5),LE.classes_))
y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))


# In[553]:


y_pred = xgb_cls.predict(X_test)
reversefactor = dict(zip(range(5), LE.classes_))
# y_test = np.vectorize(reversefactor.get)(y_test)
y_pred = np.vectorize(reversefactor.get)(y_pred)
# Making the Confusion Matrix
print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted']))


# ## Features importances

# In[554]:


sorted_importance = sorted(list(zip(X_train.columns, rf_cls.feature_importances_)), key=lambda tup: tup[1], reverse=True)
sorted_importance[:20]


# In[555]:


sorted_importance = sorted(list(zip(X_train.columns, xgb_cls.feature_importances_)), key=lambda tup: tup[1], reverse=True)
sorted_importance[:20]


# # Preprocess test csv data

# In[556]:


def process_test_csv(path):
    test_data_Singapore = pd.read_csv(path+"SingaporeTest.csv")
    test_data_NY = pd.read_csv(path+"NYTest.csv")
    test_data_London = pd.read_csv(path+"LondonTest.csv")

    test_data_Singapore['source'] = 'Singapore'
    test_data_NY['source'] = 'NY'
    test_data_London['source'] = 'London'
    test_data = pd.concat([test_data_Singapore, test_data_NY, test_data_London],ignore_index=True)

    test_data['income'] = test_data['income'].fillna("no").apply(lambda x: -1 if x == "no" else len(str(x)))

    test_data['workInfoForAgeGroupEstimation'] = test_data['workInfoForAgeGroupEstimation'].map(str).apply(process_work_info)
    test_data = test_data.join(pd.DataFrame(test_data['workInfoForAgeGroupEstimation'].tolist(), index=test_data.index, columns=['w_start_year', 'w_number_of_places', 'w_total_length', 'w_mean_length', 'w_min_length', 'w_max_length', 'w_working_now']))
    test_data.drop(columns=['workInfoForAgeGroupEstimation'], inplace=True)

    test_data['educationInfoForAgeGroupEstimation'] = test_data['educationInfoForAgeGroupEstimation'].map(str).apply(process_education_info)
    test_data = test_data.join(pd.DataFrame(test_data['educationInfoForAgeGroupEstimation'].tolist(), index=test_data.index, columns=['e_first_start_year', 'e_last_start_year', 'e_finish_year', 'e_number_of_programs']))
    test_data.drop(columns=['educationInfoForAgeGroupEstimation'], inplace=True)

    columns_to_hot = ['gender', 'relationship', 'educationLevel', 'occupation', 'source']
    test_data = pd.get_dummies(test_data, columns=columns_to_hot)
    test_data_with_labels = test_data[test_data['ageGroup'].notnull()].drop(columns=['row ID']).fillna(-1.0)
    for i in set(data_with_labels.columns)-set(test_data_with_labels.columns):
        test_data_with_labels.insert(len(test_data_with_labels.columns), i, 0)
    
    test_data_with_labels['ageGroup'] = LE.transform(test_data_with_labels['ageGroup'])
    return test_data_with_labels


# # Reading and processing new csv

# In[557]:


path = "path to the folder with test data "
# test_data_with_labels = process_test_csv(path)
# test_data_with_labels = test_data_with_labels[data_with_labels.columns]
# X, y = test_data_with_labels.drop(columns=['ageGroup']), test_data_with_labels.ageGroup


# # ## Predicting and scores
# # ### RandomForestClassifier

# # In[562]:


# get_ipython().run_cell_magic('time', '', 'y_pred_on_test_file = rf_cls.predict(X)\n\nprint("F1 score on test.csv:", f1_score(y, y_pred_on_test_file, average=\'weighted\'))\nprint("Accuracy on test.csv:", accuracy_score(y, y_pred_on_test_file))')


# # In[563]:


# reversefactor = dict(zip(range(5),LE.classes_))
# y_test = np.vectorize(reversefactor.get)(y)
# y_pred_on_test_file = np.vectorize(reversefactor.get)(y_pred_on_test_file)
# # Making the Confusion Matrix
# print(pd.crosstab(y_test, y_pred_on_test_file, rownames=['Actual'], colnames=['Predicted']))


# # ### XGBClassifier

# # In[566]:


# get_ipython().run_cell_magic('time', '', 'y_pred_on_test_file = xgb_cls.predict(X)\n\nprint("F1 score on test.csv:", f1_score(y, y_pred_on_test_file, average=\'weighted\'))\nprint("Accuracy on test.csv:", accuracy_score(y, y_pred_on_test_file))')


# # In[567]:


# reversefactor = dict(zip(range(5),LE.classes_))
# y_test = np.vectorize(reversefactor.get)(y)
# y_pred_on_test_file = np.vectorize(reversefactor.get)(y_pred_on_test_file)
# # Making the Confusion Matrix
# print(pd.crosstab(y_test, y_pred_on_test_file, rownames=['Actual'], colnames=['Predicted']))

