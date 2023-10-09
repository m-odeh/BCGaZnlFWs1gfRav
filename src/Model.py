import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#Read Data
df=pd.read_csv('C:/Users/M-ODE/Desktop/Apziva/projects/2nd Project/term-marketting/data/term-deposit-marketing-2020.csv')

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

scaler = StandardScaler()
x_numeric=df.drop(['job','marital', 'education','default','housing','loan','contact','month','y'],axis=1)
x_categorical= df.drop(["age", "balance", "day", "duration", "campaign"],axis=1)
x_numeric_scaled = pd.DataFrame(scaler.fit_transform(x_numeric), columns=x_numeric.columns) # Z-Score Scaling
#append new scaled data into one df
df_new=pd.concat([x_numeric_scaled,x_categorical],axis=1)

#SMOTE Oversampling 
X = df_new.drop('y', axis=1)
y = df_new['y']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
# Convert the resampled arrays back to DataFrames
X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
y_resampled_df = pd.DataFrame(y_resampled, columns=['y'])
#merge resampled df
df_new_final=pd.concat([X_resampled_df,y_resampled_df],axis=1)

# preparing the data 
x=df_new_final.drop(['y'], axis=1)
y=df_new_final['y'] 
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=0) # 70% training data

# xgboost
clf = xgb.XGBClassifier(n_estimators=150,max_depth=6,learning_rate=0.3,gamma=0.1,random_state=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test,y_pred))
print("F1 Score:",metrics.f1_score(y_test, y_pred))

# Use cross-validation to evaluate the model
scores=cross_val_score(clf, X_train,y_train, scoring="f1",cv=5)
print("Cross-validation scores (f1 Score):", scores)

print("Mean F1 Score (cross validation):", np.mean(scores))

pickle.dump(clf, open('model.pkl', 'wb'))