import numpy as np
import pandas as pd
import os
import pickle
import yaml
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

with open("params.yaml","r") as file:
    params=yaml.safe_load(file)
print(params)
n_estimators=params['model']['n_estimators']
max_depth=params['model']['max_depth']
test_size=params['data']['test_size']

train_data=pd.read_csv("E:\\air_quality_pipeline\\data\\processed\\processed_train.csv")
x_train=train_data.drop('Air Quality',axis=1)
y_train=train_data['Air Quality']
encoder=LabelEncoder()
y_train_encode=encoder.fit_transform(y_train)
model=RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth)
model.fit(x_train,y_train_encode)
pickle.dump(model,open("models/model.pkl","wb"))
