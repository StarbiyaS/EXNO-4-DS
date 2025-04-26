# EXNO:4-DS
# DATE: 26/04/25
# NAME: STARBIYA S
# REG NO: 212223040208
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```
![image](https://github.com/user-attachments/assets/6b5cf5eb-dbae-426d-acc9-424c4bdb8db0)

```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/9e3ff961-2242-452d-a898-4d61f1f3f893)
```
df.dropna()
```
![image](https://github.com/user-attachments/assets/b47113f8-24c5-4a7e-928a-de04374ad106)
```
max_vals=np.max(np.abs(df[['Height','Weight']]),axis=0)
max_vals
```
![image](https://github.com/user-attachments/assets/dcbd999a-aa0d-408c-b8a7-db78a2375ff7)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/ceef36e1-7d4e-4962-b2fc-a0e73905fac7)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![image](https://github.com/user-attachments/assets/da0699d4-7e88-467c-b1a4-b8d6c1246d2a)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3=pd.read_csv("/content/bmi.csv")
df3.head()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```
![image](https://github.com/user-attachments/assets/5201721b-6420-4093-bc45-0e79167412cc)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4=pd.read_csv("/content/bmi.csv")
df4.head()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```
![image](https://github.com/user-attachments/assets/3dae8aaf-aae3-43d8-a28e-240a3c2ffc26)
```
import pandas as pd
df=pd.read_csv("/content/income(1) (1).csv")
df.info()
```
![image](https://github.com/user-attachments/assets/5de3bf68-56ae-4ace-bdfd-88c21889e92d)
```
df
```
![image](https://github.com/user-attachments/assets/53b80b8d-effd-438d-9391-df4bdb8bd3c7)
```
df_null_sum=df.isnull().sum()
df_null_sum
```
![image](https://github.com/user-attachments/assets/d12e6d38-e2c9-4bd1-80df-a44bae4baaf2)
```
categorical_columns=['JobType','EdType','maritalstatus','occupation','relationship','race','gender','nativecountry']
df[categorical_columns]=df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/bea8319e-44db-453e-b8ff-aaccb1df2065)
```
df[categorical_columns]=df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/753f39b7-d580-4321-ab53-b994b7ee519c)
```
x=df.drop(columns=['SalStat'])
y=df['SalStat']
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
rf=RandomForestClassifier()
rf.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/51f958be-d1a1-428a-8206-43904c6ae092)
```
y_pred=rf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/d472655d-9e14-43b6-8eaf-a1edc1301902)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,chi2,f_classif
categorical_columns=['JobType','EdType','maritalstatus','occupation','relationship','race','gender','nativecountry']
df[categorical_columns]=df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/7eb46c4a-da49-4bc0-9ad0-89e9983080be)
```
df[categorical_columns]=df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
```
x=df.drop(columns=['SalStat'])
y=df['SalStat']
k_chi2=6
selector_chi2=SelectKBest(score_func=chi2,k=k_chi2)
x_chi2=selector_chi2.fit_transform(x,y)
selected_features_chi2=x.columns[selector_chi2.get_support()]
print("Selected features using chi-squared test:")
print(selected_features_chi2)
```
![image](https://github.com/user-attachments/assets/ae6becd6-63cf-41a6-aafd-3d68ea1b24ce)
```
selected_features=['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss','hoursperweek']
x=df[selected_features]
y=df['SalStat']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
rf=RandomForestClassifier(n_estimators=100,random_state=42)
rf.fit(x_train,y_train)
```
![image](https://github.com/user-attachments/assets/f81cc737-dba8-4e90-aa99-de18d6f53df2)
```
y_pred=rf.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(f"Model accuracy using selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/d1d51a0f-2583-4442-aa66-1082d7805ec0)
```
# @title
!pip install skfeature-chappers
```
![image](https://github.com/user-attachments/assets/c8d01bdd-422e-4314-96de-cc5f5b46cfa6)
```
# @title
import numpy as np
import pandas as pd
from skfeature.function.similarity_based import fisher_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# @title
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/d5fb4b7f-318a-4412-9221-591d99564d7e)
```
# @title
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
# @title
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/3a99c23f-f04d-42bb-b2ff-b571d013aaa3)
```
k_anova = 5
selector_anova = SelectKBest(score_func=f_classif, k=k_anova)
x_anova = selector_anova.fit_transform(x, y)
selected_features_anova = x.columns[selector_anova.get_support()]
print("\nSelected features using ANOVA:")
print(selected_features_anova)
```
![image](https://github.com/user-attachments/assets/756f639e-e7c2-4ed5-9c5c-b9ccbade1b49)
```
import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
df[categorical_columns] = df[categorical_columns].astype('category')
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/11f41954-1c84-4acb-a58e-bcd021cf0c09)
```
df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
df[categorical_columns]
```
![image](https://github.com/user-attachments/assets/f070c85f-aa16-4687-806d-174665ff5166)
```
X = df.drop(columns=['SalStat'])
y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select = 6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```
![image](https://github.com/user-attachments/assets/3477eb24-956b-4722-bf8b-27e61c386d63)
![image](https://github.com/user-attachments/assets/1795f83a-1cb6-4ebf-958e-58a8ce736453)
```
selected_features = X.columns[rfe.support_]
print("Selected features using RFE:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/fb33c3ee-5f30-409a-903c-872e764b7c80)
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
X_selected = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy using Fisher Score selected features: {accuracy}")
```
![image](https://github.com/user-attachments/assets/86bfd9f1-d968-4ffc-ad52-905d56277647)

# RESULT:
Thus,Feature selection and Feature scaling has been used on the given dataset.
