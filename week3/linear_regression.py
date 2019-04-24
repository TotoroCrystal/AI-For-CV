import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

USAhousing = pd.read_csv('USA_Housing.csv')
USAhousing.head()
USAhousing.info()
USAhousing.describe()

# consider Price as the dependent variable, and the rest as independent variables
# predict Price
sns.pairplot(USAhousing)
# sns.distplot(USAhousing['Price'])
# sns.scatterplot(x = 'Avg. Area Income', y = 'Price', data = USAhousing)
plt.show()

sns.heatmap(USAhousing.corr(),
        xticklabels = USAhousing.corr().columns,
        yticklabels = USAhousing.corr().columns)
plt.show()

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
Y = USAhousing['Price']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 0)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

lr = LinearRegression()
lr.fit(X_train, Y_train)
Y_predict = lr.predict(X_test)

print("accuracy：", lr.score(X_test, Y_test))

mse = np.average((Y_predict - np.array(Y_test)) ** 2)
rmse = np.sqrt(mse)
print("MSE: ", mse)
print("RMSE: ", rmse)

#print(" House price coefficients: ", lr.coef_)
print("The intercept for our model is {}".format(lr.intercept_))
print(lr.coef_)
print(X.columns)

coeff_df = pd.DataFrame(lr.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_predict})
print(df)