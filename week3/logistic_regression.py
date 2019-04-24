import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import seaborn as sns

Diabetes = pd.read_csv('diabetes.csv')

X = Diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
Y = Diabetes['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)
X_train = X_train.astype(np.float64)
X_test = X_test.astype(np.float64)

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

logit = LogisticRegression(solver = 'lbfgs')
logit.fit(X_train, Y_train)
Y_predict = logit.predict(X_test)

print('accuracy: ', logit.score(X_test, Y_test))
print('intercept: ', logit.intercept_)

print(Diabetes.info())
print(Diabetes.columns)
print(Diabetes.shape)
print(Diabetes.describe())
print(logit.coef_)
print(X.columns)

coeff = pd.DataFrame(np.array(logit.coef_).reshape(8,1), X.columns, columns=['Coefficient'])
print(coeff)

# df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_predict})
# print(df)

# confusion matrix to evaluation
cnf_matrix = metrics.confusion_matrix(Y_test, Y_predict)
print(cnf_matrix)

# visualizing confusion matrix using heatmap
class_names = [0, 1]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'YlGnBu', fmt = 'g')
ax.xaxis.set_label_position('top')
plt.tight_layout()
plt.title('Confusion matrix', y = 1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()
# confusion matrix evaluation
print('Accuracy:', metrics.accuracy_score(Y_test, Y_predict))
print('Precision:', metrics.precision_score(Y_test, Y_predict))
print('Recall:', metrics.recall_score(Y_test, Y_predict))
# ROC Curve: Receiver Operating Characteristic(ROC) curve is a plot of the true positive rate against the false positive rate. It shows the tradeoff between sensitivity and specificity.
Y_predict_proba = logit.predict_proba(X_test)[::, 1]
fpr, tpr, _ = metrics.roc_curve(Y_test, Y_predict_proba)
auc = metrics.roc_auc_score(Y_test, Y_predict_proba)
plt.plot(fpr, tpr, label = 'data 1, auc = ' + str(auc))
plt.legend(loc = 4)
plt.show()