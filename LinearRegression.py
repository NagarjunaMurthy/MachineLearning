import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Salary_Data.csv")
X=dataset.iloc[:,:-1]
Y=dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( X,Y, test_size=0.25)

from sklearn.linear_model import LinearRegression
regression=LinearRegression()

regression.fit(x_train,y_train)

y_pred=regression.predict(x_test)

plt.scatter(x=x_train,y=y_train,color='red')
plt.plot(x_train,regression.predict(x_train),color="blue")
plt.title('Salary Vs Expirence')
plt.xlabel("Years of Expierence")
plt.ylabel("Salary")
plt.show()

plt.scatter(x=x_test,y=y_test,color='red')
plt.plot(x_train,regression.predict(x_train),color="blue")
plt.title('Salary Vs Expirence')
plt.xlabel("Years of Expierence")
plt.ylabel("Salary")
plt.show()

regression.score(x_test,y_test)*100
