# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 18:50:57 2022

@author: siddhardhan
"""

from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
classifier = svm.SVC(kernel='linear',probability=True)

# loading the diabetes dataset to a pandas DataFrame
diabetes_dataset = pd.read_csv('heart_disease_data.csv') 

X = diabetes_dataset.drop(columns = 'target', axis=1)
Y = diabetes_dataset['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
classifier.fit(X_train, Y_train)

X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print (training_data_accuracy)
print (X)
print (Y)




app = FastAPI()

class model_input(BaseModel):
    
    age : int
    sex : int
    cp : int
    trestbps : int
    chol : int
    fbs : int
    restecg : int
    thalach : int
    exang : int
    oldpeak : float
    slope : int
    ca : int
    thal : int       
        
# loading the saved model

@app.post('/classifier')
def diabetes_predd(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    age = input_dictionary['age']
    sex = input_dictionary['sex']
    cp = input_dictionary['cp']
    chol = input_dictionary['chol']
    fbs = input_dictionary['fbs']
    restecg = input_dictionary['restecg']
    thalach = input_dictionary['thalach']
    exang = input_dictionary['exang']
    oldpeak = input_dictionary['oldpeak']
    slope = input_dictionary['slope']
    ca = input_dictionary['ca']
    thal = input_dictionary['thal']
    
    
    input_list = [age, sex, cp, chol, fbs, restecg, thalach, exang,oldpeak,slope,ca,thal]
    
    prediction = classifier.predict_proba([input_list])
    
    if (prediction[0][0] >= 0.75):
        return 2
    elif (prediction[0][0] < 0.75 and prediction[0][0] >= 0.25):
        return 1
    else:
        return 0
