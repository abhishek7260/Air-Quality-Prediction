import pandas as pd
import numpy as np
import os 
train_file = "./data/raw/train.csv"
test_file = "./data/raw/test.csv"

train_data=pd.read_csv(train_file)
test_data=pd.read_csv(test_file)

def remove_outlier(df,columns=None):
    if columns is None:
        columns=df.select_dtypes(include=['number']).columns.tolist()
        
    for col in columns:
        q1=df[col].quantile(.25)    
        q3=df[col].quantile(.75)
        iqr=q3-q1
        
        lower_bound=q1-1.5*iqr    
        upper_bound=q3+1.5*iqr    
        df=df[(df[col]>=lower_bound) & (df[col]<=upper_bound)]
    return df    
processed_train_data=remove_outlier(train_data)
processed_test_data=remove_outlier(test_data)
data_path=os.path.join("data","processed")
os.makedirs(data_path,exist_ok=True)
processed_train_data.to_csv(os.path.join(data_path,"processed_train.csv"),index=False)
processed_test_data.to_csv(os.path.join(data_path,"processed_test.csv"),index=False)
        
    