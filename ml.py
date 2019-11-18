# ml algos http://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/
from sklearn.ensemble import *
from sklearn.datasets import *
from sklearn.svm import *
from sklearn.linear_model import *
from sklearn.neural_network import *
from sklearn.kernel_ridge import *

d=load_diabetes()
#d=load_boston()

X=d.data
y=d.target
#savetxt('boston',X)

models=[ 
AdaBoostRegressor(),
BaggingRegressor(),
ElasticNet(), 
ExtraTreesRegressor(),
GradientBoostingRegressor(), 
KernelRidge(),
Lasso(), 
LinearRegression(), 
MLPRegressor(max_iter=10000),  
RandomForestRegressor(n_estimators=10), 
SVR(kernel='linear'), 
SVR(kernel='rbf'),
]

for est in models:
	est.fit(X,y)
	print str(est).split('(')[0]+'\t\t'+str(est.score(X,y))
