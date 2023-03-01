import numpy as np 
import time
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression


cmap=plt.get_cmap('viridis')
#######################################################################
head=['age', 'z', 'VR',  'VT',  'VZ', 'Vtot']
lab= [r"$\rm{Age}$", r"$\rm{Height}$", r"$V_{\rm{R}}$", r"$V_{\rm{T}}$", r"$V_{\rm{Z}}$", r"$V_{\rm{tot}}$" ]


f1=open("./files/iqrs.txt","r")
nm=sum(1 for line in f1) 
par=np.zeros(( nm, 6 )) 
par=np.loadtxt("./files/iqrs.txt")
df= pd.DataFrame(par, columns = head)
fif=open("./files/KFoldRegression.txt","w")
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


############################################################################
ns=int(10)
for i in range(4): 
    result=np.zeros((5))

    #model = LassoCV(alphas=None,cv=10, random_state=450, max_iter=100000)
    #model = RidgeCV(alphas=[0.0001,0.01,0.1,0.02,0.03,0.1,0.2,0.3,0.4,0.9])##,cv=10)##, random_state=450, max_iter=100000)
    model= LinearRegression()
    
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
        #print(i, r2s,  mape,   mse,   rmse)
        ni+=1
    result=np.array([ np.mean(scores[:,0]),  np.mean(scores[:,1]),  np.mean(scores[:,2]),  np.mean(scores[:,3]),  np.mean(scores[:,4]) ])

    fif=open("./files/KFoldRegression.txt","a+")
    np.savetxt(fif, result.reshape((1,5)),fmt="KFOLD $%d$  & $%.3f$ &  $%.3f$  &  $%.3f$  &  $%.3f$") 
    np.savetxt(fif, model.coef_.reshape((1,2)),fmt="COEF $%.7f$   &    $%.7f$") 
    fif.write("\n***************************************\n")
    fif.close()
    
    print( "Output: ", result)
    #print ("Alpha:  ",   model.alpha_ )
    print ("coefficents:   ",   model.coef_ ,  len(model.coef_ ))
    #print ("coefficents_sparse:   ", np.array(df.columns)[model.coef_==0])
    print ("intercept:     ",   model.intercept_)
    print( "********************************************************" )

###################################################################   


