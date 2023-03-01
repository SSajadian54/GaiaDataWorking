import numpy as np 
import time
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

cmap=plt.get_cmap('viridis')
#######################################################################
head=['age', 'z', 'VR',  'VT',  'VZ', 'Vtot']
lab= [r"$\rm{Age}$", r"$\rm{Height}$", r"$V_{\rm{R}}$", r"$V_{\rm{T}}$", r"$V_{\rm{Z}}$", r"$V_{\rm{tot}}$" ]


f1=open("./iqrs.txt","r")
nm=sum(1 for line in f1) 
par=np.zeros(( nm,6 )) 
par=np.loadtxt("./iqrs.txt")
df= pd.DataFrame(par, columns = head)
fif=open("./KFoldSVR.txt","w")
fif.close()

print("describe:     ",  df.describe())
print("Columns:  ",  df.columns, "len(columns):  ",  len(df.columns) )
print("******************************************************")
#######################################################################

x=np.zeros(( nm , 2))
y=np.zeros(( nm , 4))
for i in range(nm):  
    x[i,0], x[i,1]=                 df.age[i], df.z[i] 
    y[i,0], y[i,1], y[i,2], y[i,3]= df.VR[i], df.VT[i], df.VZ[i], df.Vtot[i] 
print ("Feature,  output:  ",  x, y)    

###########################################################################

ns=int(10)
for i in range(4): 
    result=np.zeros((5))
    
    model = SVR(kernel = "poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)    
    kf = KFold(n_splits=ns, shuffle=True, random_state=42)
    kf.get_n_splits(x)
    scores = np.zeros((ns, 5)); ni=0
    for train, test in kf.split(x):
        xtrain, xtest = x[train,:], x[test,:]
        ytrain, ytest = y[train,i], y[test,i]
        model.fit(xtrain, ytrain )
        ypred =  model.predict(xtest)
       
        r2s=   np.abs(1.0- np.sum((ytest-ypred)**2.0)/np.sum((ytest-np.mean(ytest))**2.0))
        mape=  np.abs(np.mean(np.abs((ytest-ypred)/ytest )))*100.0
        mse=   np.mean( (ytest-ypred)**2.0 )
        rmse=  np.sqrt(mse)   
        scores[ni,:]=i, r2s, mape, mse, rmse
        ni+=1
    result=np.array([ np.mean(scores[:,0]),np.mean(scores[:,1]),np.mean(scores[:,2]),np.mean(scores[:,3]),np.mean(scores[:,4]) ])     
    fif=open("./KFoldSVR.txt","a+")
    np.savetxt(fif,res.reshape((-1,5)),fmt="KFOLD $%d$ & $%.3f$ &  $%.3f$  &  $%.3f$  &  $%.3f$") 
    fif.write("\n***************************************\n")
    fif.close()
    
    print( "Output: ", result)
    print ("Alpha:  ",   model.alpha_ )
    print ("coefficents:   ",   model.coef_ ,  len(model.coef_ ))
    #print ("coefficents_sparse:   ", np.array(df.columns)[model.coef_==0])
    print ("intercept:     ",   model.intercept_)
    print( "********************************************************" )

################################################################### 
