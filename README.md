# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and read the dataframe.
2. Assign hours to X and score to Y.
3. Implement training set and test set of the dataframe.
4. Plot the required graph both for test data and training data.
5. Find the values of MSE,MAE and RMSE.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MARELLA HASINI
RegisterNumber:  212223240083
*/
i)import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("student_scores.csv")
df.head()
ii) df.tail()
iii)X=df.iloc[:,:-1].value
    X
iv)
Y=df.iloc[:,1].values
Y
v)
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)
Y_pred
vi)
Y_test
vii)
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Sort)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
viii)
plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Sort)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
ix) 
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)
x)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
xi)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)

i)![OUTPUT](head.png)
ii)![OUTPUT](tail.png)
iii)![OUTPUT](<X array.png>)
iv)![OUTPUT](<Y array.png>)
v)![OUTPUT](Y_pred.png)
vi)![OUTPUT](Y_test.png)
vii)![OUTPUT](<Train sort.png>)
viii)![OUTPUT](<Test Sort.png>)
xi)![OUTPUT](MSE.png)
x)![OUTPUT](MAE.png)
xi)![OUTPUT](RMSE.png)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
