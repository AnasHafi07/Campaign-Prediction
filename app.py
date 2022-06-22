# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 14:11:38 2022

@author: ANAS
"""

import os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder

#%% Statics

CSV_TEST_PATH = os.path.join(os.getcwd(),'Datasets', 'Test.csv')
SAVE_RESULT = os.path.join(os.getcwd(), 'Datasets',
                           'new_customers_results.csv')
SCALER_PATH = os.path.join(os.getcwd(), 'Objects', 'scaler.pkl')
OHE_PATH = os.path.join(os.getcwd(), 'Objects', 'ohe.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'Objects', 'knn.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'Objects', 'dl_model.h5')

ID_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','id.pkl')
JT_ENCODER_PATH_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','job_type.pkl')
MARITAL_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','marital.pkl')
EDUCATION_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','education.pkl')
DEFAULT_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','default.pkl')
HL_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','housing_loan.pkl')
PL_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','personal_loan.pkl')
COMMUNICATION_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','communication_type.pkl')
MONTH_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','month.pkl')
PCO_ENCODER_PATH = os.path.join(os.getcwd(),'Objects','prev_campaign_outcome.pkl')


#%% Functions
def return_saved_objects(path):
    """
    Return an object from a pickle file.
    Parameters
    ----------
    path : str
        The path to the saved objects.
    Returns
    -------
    object
        Return an object from a pickle file.
    """
    with open(path, 'rb') as file:
        return pickle.load(file)
    
#%%
ss_scaler = return_saved_objects(SCALER_PATH)
oh_encoder = return_saved_objects(OHE_PATH)
imputer = return_saved_objects(KNN_PATH)

ID = return_saved_objects(ID_ENCODER_PATH)
JT = return_saved_objects(JT_ENCODER_PATH_ENCODER_PATH)
EDUCATION = return_saved_objects(EDUCATION_ENCODER_PATH)
DEFAULT = return_saved_objects(DEFAULT_ENCODER_PATH)
HL = return_saved_objects(HL_ENCODER_PATH)
PL = return_saved_objects(PL_ENCODER_PATH)
COMMUNICATION = return_saved_objects(COMMUNICATION_ENCODER_PATH)
MONTH = return_saved_objects(MONTH_ENCODER_PATH)
PCO = return_saved_objects(PCO_ENCODER_PATH)


# %% Load DL model
model = load_model(MODEL_PATH)

# %% Load test data
df_test = pd.read_csv(CSV_TEST_PATH)

cater_columns = df_test.columns[df_test.dtypes =='object']

con_columns = df_test.columns[(df_test.dtypes == 'int64') | (df_test.dtypes == 'float64')]

#%% data cleaning

df_dummy = df_test.copy()

le = LabelEncoder()

paths = [ID_ENCODER_PATH,JT_ENCODER_PATH_ENCODER_PATH, MARITAL_ENCODER_PATH,
         EDUCATION_ENCODER_PATH, DEFAULT_ENCODER_PATH, HL_ENCODER_PATH,PL_ENCODER_PATH,
         COMMUNICATION_ENCODER_PATH, MONTH_ENCODER_PATH,PCO_ENCODER_PATH]

for index,i in enumerate(cater_columns):
    temp = df_dummy[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df_dummy[i] = pd.to_numeric(temp,errors = 'coerce')
    with open(paths[index],'wb') as file:
        pickle.dump(le,file)

imputed = imputer.transform(df_dummy)

df_dummy['days_since_prev_campaign_contact'] = np.floor(df_dummy['days_since_prev_campaign_contact'])
df_dummy = df_dummy.drop(labels=['id','month','days_since_prev_campaign_contact'], axis=1)


X = df_dummy.drop(labels=['prev_campaign_outcome'], axis=1)
y = df_dummy['prev_campaign_outcome']

X_scaled = ss_scaler.transform(X)

y_pred = model.predict(X_scaled).argmax(1)

label_pred = oh_encoder.inverse_transform(np.expand_dims(y_pred,axis=-1))

df_test['prev_campaign_outcome'] = label_pred
df_test.to_csv(SAVE_RESULT, index=False)


