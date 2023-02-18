import numpy as np 
import time
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from sklearn import tree
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn import datasets, linear_model
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text
cmap=plt.get_cmap('viridis')
#######################################################################
head=['age', 'z', 'b', 'VR',  'VT',  'VZ', 'Vtot', 'num']
lab= [r"$\rm{Age}$", r"$\rm{Height}$", r"$b$", r"$V_{\rm{R}}$", r"$V_{\rm{T}}$", r"$V_{\rm{Z}}$", r"$V_{\rm{tot}}$" ]


#df= pd.read_csv("./iqrs.txt",sep="  ",skipinitialspace = True, header=None, usecols=[0,1,2,3,4,5,6,7], names=head)
f1=open("./iqrs.txt","r")
nm=sum(1 for line in f1) 
par=np.zeros(( nm,8 )) 
par=np.loadtxt("./iqrs.txt")
df= pd.DataFrame(par, columns = head)
fif=open("./KFoldLRegression.txt","w")
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
    res=np.zeros((ns+1,6))
    model = linear_model.LinearRegression()##random_state=89)
    #model=RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=89)
    cv = model_selection.KFold(n_splits=ns)
    nkf=0; 
    for traini, testi in cv.split(x):
        xtrain=np.zeros((len(traini),2))    
        xtest= np.zeros((len(testi), 2))          
        ytrain=np.zeros((len(traini) ))   
        ytest= np.zeros((len(testi)  ))
        for j in range(len(traini)):
            k=int(traini[j]) 
            xtrain[j,:]= x[k,:] 
            ytrain[j]  = y[k,i]
        for j in range(len(testi)):
            k=int(testi[j])
            xtest[j,:]= x[k,:]
            ytest[j]  = y[k,i]
        model.fit(xtrain, ytrain)
        ypred = model.predict(xtest)
        r2s=   metrics.r2_score(ytest, ypred)
        mape=  np.abs(np.mean(np.abs((ytest-ypred)/ytest)))*100.0
        mse=   metrics.mean_squared_error(ytest , ypred)
        rmse=  np.sqrt(mse)    
        res[nkf,:] =i, nkf,  r2s,  mape,   mse,   rmse 
        nkf+=1
        
    res[ns,:]= i, ns ,np.mean(res[:ns,2]), np.mean(res[:ns,3]), np.mean(res[:ns,4]), np.mean(res[:ns,5])    
    fif=open("./KFoldLRegression.txt","a+")
    np.savetxt(fif,res.reshape((-1,6)),fmt="KFOLD $%d$  & $%d$  & $%.3f$ &  $%.3f$  &  $%.3f$  &  $%.3f$") 
    #np.savetxt(fif, model.feature_importances_.reshape((-1,2)),  fmt="IMPORT   $%.5f$  &   $%.5f$")
    fif.write("\n***************************************\n")
    fif.close()
    print( "Output: ", res[10,:] )
    #print( "importance_i: ", model.feature_importances_ )
    print( "********************************************************" )

###################################################################   
'''
model=RandomForestRegressor(n_estimators=50, max_depth=4, n_jobs=1, random_state=89)
model.fit(xtrain, ytrain)
fig=plt.figure(figsize=(20,20))
_=tree.plot_tree(model.estimators_[0], feature_names = lab[:2],filled = True);
fig.savefig("./Histos/OneTree.jpg", dpi=200)
print( "ONE tree was plotted ******************************" )
'''
###################################################################   
'''
start_time =time.time()
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
elapsed_time = time.time() - start_time
print("Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances = pd.Series(importances, index=lab[:2])
plt.clf()
fig= plt.figure(figsize=(8,6))
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_ylabel(r"$\rm{Feature}~\rm{importance}$", fontsize=18)
plt.xticks(fontsize=16, rotation=0)
plt.yticks(fontsize=16, rotation=0)
plt.grid(axis = 'y', color = '#D3D3D3', linestyle = 'dashed')
fig.tight_layout()
fig.savefig("./Histos/import1a.jpg", dpi=200)
'''
###################################################################  
    

array=np.zeros((len(xtest) , 3))
for i in range(len(xtest)): 
    array[i,0]= float(xtest[i,0])
    array[i,1]= float(ytest[i ])
    array[i,2]= float(ypred[i ])
plt.clf()
fig= plt.figure(figsize=(8,6))
plt.scatter(array[:,0], array[:,1],c="m", s=3.0, label="original")
plt.scatter(array[:,0], array[:,2],c="b", s=3.5, label="predicted")
plt.title("y-test and y-predicted data")
plt.xlabel(r"$Age(Gyr)$", fontsize=18)
plt.ylabel(r"$V_{\rm{tot}}(km/s)$", fontsize=18)
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.grid(linestyle='dashed')
plt.savefig("./Histos/ExampleVtot.jpg",  dpi=200) 
##################################################################






