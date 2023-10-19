import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the pickled model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load new data (assuming it's in a CSV file)
df = pd.read_csv('C:/Users/M-ODE/Desktop/Apziva/projects/2nd Project/term-marketting/data/term-deposit-marketing-2020.csv')

# data preprocessing 
#Treating missing values
columns_with_unknown = df.columns[df.isin(["unknown"]).any()]     # checking null/unknown columns
df[columns_with_unknown] = df[columns_with_unknown].replace("unknown", pd.NA)  # replace unknown with NA
# Mode imputation for categorical columns
for column in columns_with_unknown:
    mode_value = df[column].mode()[0]  # Get the mode (most frequent value)
    df[column] = df[column].fillna(mode_value)

# Label encode the categorical variable 
df['job'] = pd.factorize(df['job'])[0]
df['marital'] = pd.factorize(df['marital'])[0]
df['education'] = pd.factorize(df['education'])[0]
df['default'] = pd.factorize(df['housing'])[0]
df['loan'] = pd.factorize(df['loan'])[0]
df['contact'] = pd.factorize(df['contact'])[0]
df['month'] = pd.factorize(df['month'])[0]
df['y'] = pd.factorize(df['y'])[0]
df['housing'] = pd.factorize(df['housing'])[0]

# Scaling 
scaler = StandardScaler()
x_numeric=df.drop(['job','marital', 'education','default','housing','loan','contact','month','y'],axis=1)
x_categorical= df.drop(["age", "balance", "day", "duration", "campaign"],axis=1)
x_numeric_scaled = pd.DataFrame(scaler.fit_transform(x_numeric), columns=x_numeric.columns) # Z-Score Scaling
#append new scaled data into one df
new_data=pd.concat([x_numeric_scaled,x_categorical],axis=1)


# Make predictions using the loaded model
predictions = model.predict(new_data)

# You can save the predictions to a CSV file if needed
predictions_df = pd.DataFrame(predictions, columns=['Predicted_Label'])
predictions_df.to_csv('predictions.csv', index=False)

print('Predictions saved to predictions.csv')
