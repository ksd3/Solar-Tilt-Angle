from numpy import loadtxt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from numpy import absolute
from pandas import read_csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot
import itertools

vc1 = pd.read_csv('rooffull.csv')
vc2 = pd.read_csv('canopyfull.csv')
vc3 = pd.read_csv('groundfull.csv')
vtnt=pd.concat([vc1,vc2,vc3],axis=0)
vtnt.fillna(0)
X = vtnt.iloc[:, 0:-1]
y = vtnt.iloc[:, -1]
X = X.values
y = y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,shuffle=False)
model = MLP()
eval_set = [(X_train, y_train), (X_test, y_test)]
model.fit(X_train, y_train, verbose=False)
y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error
rms = mean_squared_error(y_test, y_pred, squared=False)
results = model.evals_result()
epochs = len(results['validation_1']['rmse'])
x_axis = range(0, epochs)
results = model.evals_result()
epochs = len(results['validation_1']['rmse'])
x_axis = range(0, epochs)
# plot log loss
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('%Loss')
pyplot.title('Loss for MLP')
pyplot.show()
fig, ax = pyplot.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
pyplot.ylabel('Regression Error')
pyplot.title('MLP Regression Error')
pyplot.show()
for i in range(0,90):
    X_test[:,-2]=i
    y_pred = model.predict(X_test)
    sum=np.sum(y_pred, dtype = np.float32)
    
    if sum>a:
        a=sum
        angle=i
        print(a)
        print(i)

def seq(start, end, step):
    if step == 0:
        raise ValueError("0 step")
    sample_count = int(abs(end - start) / step)
    return itertools.islice(itertools.count(start, step), sample_count)
for j in seq(angle-1,angle+1,0.1):
X_test[:,-2]=j
y_pred = model.predict(X_test)
sum=np.sum(y_pred, dtype = np.float32)
if sum>a:
    a=sum
    print(a)
    print(i)
  results['validation_0']['rmse']