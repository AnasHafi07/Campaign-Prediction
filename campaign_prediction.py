# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 09:03:13 2022

@author: ANAS
"""

#%% Imports

import os 
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#from modules_for_campaign_prediction import EDA
from tensorflow.keras.utils import plot_model

#%% Statics

CSV_PATH = os.path.join(os.getcwd(),'Datasets', 'Train.csv')
STATICS_PATH = os.path.join(os.getcwd(),'Statics')

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

SCALER_PATH = os.path.join(os.getcwd(), 'Objects', 'scaler.pkl')
OHE_PATH = os.path.join(os.getcwd(), 'Objects', 'ohe.pkl')
KNN_PATH = os.path.join(os.getcwd(), 'Objects', 'knn.pkl')
MODEL_PATH = os.path.join(os.getcwd(), 'Objects', 'dl_model.h5')
PLOT_MODEL_PATH = os.path.join(os.getcwd(), 'Statics',
                               'model-architecture.png')

COLUMN_NAMES = ['id', 'customer_age', 'job_type', 'marital', 'education', 'default',
       'balance', 'housing_loan', 'personal_loan', 'communication_type',
       'day_of_month', 'month', 'last_contact_duration',
       'num_contacts_in_campaign', 'days_since_prev_campaign_contact',
       'num_contacts_prev_campaign', 'prev_campaign_outcome',
       'term_deposit_subscribed']

#%% Functions

#%% EDA

#%% Step 1) Data Loading

df = pd.read_csv(CSV_PATH)

#%%  Step 2) Data inspection
df.info()
stats = df.describe().T
"""
    Decided to drop id, month and days_since_prev_campaign_contact due to 
    not having relationship or too many NaNs
    
    For object type, need to do label encodeing for categorical 
"""
df.columns

cater_columns = df.columns[df.dtypes =='object']

con_columns = df.columns[(df.dtypes == 'int64') | (df.dtypes == 'float64')]


print('\nMissing values:\n', df.isna().sum()) 

df.duplicated().sum()

#%% Step 3) Data Cleaning 

df_dummy = df.copy()

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
        
print(df_dummy.isna().sum())

df_dummy = df_dummy.drop(labels=['term_deposit_subscribed'], axis=1)

from sklearn.impute import KNNImputer

knn_imp = KNNImputer()
df_dummy = knn_imp.fit_transform(df_dummy)
df_dummy = pd.DataFrame(df_dummy)

with open(KNN_PATH, 'wb') as file:
    pickle.dump(knn_imp,file)

print(df_dummy.isna().sum())

df_dummy = df_dummy.rename(columns={0: 'id', 1:'customer_age',2: 'job_type',
                                    3: 'marital', 4:'education', 5:'default',
        6:'balance',7: 'housing_loan',8: 'personal_loan',9: 'communication_type',
        10:'day_of_month',11: 'month', 12:'last_contact_duration',
        13:'num_contacts_in_campaign', 14:'days_since_prev_campaign_contact',
        15:'num_contacts_prev_campaign',16: 'prev_campaign_outcome'})


df_dummy['days_since_prev_campaign_contact'] = np.floor(df_dummy['days_since_prev_campaign_contact'])
df_dummy = df_dummy.drop(labels=['id','month','days_since_prev_campaign_contact'], axis=1)

#%% STEP 4 Features selection

# Skip first but then you will never reach the wanted

#%% Step 5) Pre-processing

X = df_dummy.drop(labels=['prev_campaign_outcome'], axis=1)
y = df_dummy['prev_campaign_outcome']

nb_class = len(np.unique(y))

# Input 14
# Output 1

from sklearn.preprocessing import StandardScaler
sts = StandardScaler()
X = sts.fit_transform(X)

with open(SCALER_PATH, 'wb') as file:
    pickle.dump(sts,file)

from sklearn.preprocessing import OneHotEncoder
import numpy as np

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=1))

with open(OHE_PATH, 'wb') as file:
    pickle.dump(ohe,file)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3,
                                                    random_state=123)

#%% Model development

from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential, Input

model = Sequential ()
model.add(Input(shape=(13,))) #need to do 11, since need tuple, from X shape
model.add(Dense(32, activation = 'relu', name ='Hidden_Layer_1'))
# no need flatten since we're not dealing with 2D (image)
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32, activation = 'relu', name ='Hidden_Layer_2'))
# no need flatten since we're not dealing with 2D (image)
model.add(BatchNormalization())
model.add(Dropout(0.2))
# buat satu dulu then tambah if not enough
model.add(Dense(4,activation='softmax', name='Output_layer'))
 #softmax since classification
model.summary() # to visualize

model.compile(loss = 'categorical_crossentropy', optimizer='adam',
                   metrics=['acc'])

from datetime import datetime
from tensorflow.keras.callbacks import TensorBoard
LOG_PATH = os.path.join(os.getcwd(), 'Logs')
log_dir = os.path.join(LOG_PATH, datetime.now().strftime('%Y%m%d-%H%M%S'))
tb_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

from tensorflow.keras.callbacks import EarlyStopping
early_stopping_callback = EarlyStopping(monitor='loss', patience =3)

hist = model.fit(X_train,y_train,batch_size=64,epochs=5,
                      validation_data=(X_test,y_test),
                      callbacks= [early_stopping_callback, tb_callback])

plot_model(model, to_file=PLOT_MODEL_PATH)
model.save(MODEL_PATH)

#%%
hist.history.keys()

training_loss = hist.history['loss']
training_acc = hist.history['acc']
validation_acc = hist.history['val_acc']
validation_loss = hist.history['val_loss']

plt.figure()
plt.plot(training_loss)
plt.plot(validation_loss)
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.figure()
plt.plot(training_acc)
plt.plot(validation_acc)
plt.legend(['train_acc', 'val_acc'])
plt.show()

#%%
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay

results = model.evaluate(X_test,y_test)
print(results)

pred_y = np.argmax(model.predict(X_test),axis=1)
true_y = np.argmax(y_test,axis=1)

cm = confusion_matrix(true_y, pred_y)
cr = classification_report(true_y, pred_y)
print("\n",cm) 
print("\n",cr)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()
