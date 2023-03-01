import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from scipy.stats import iqr
cmap=plt.get_cmap('viridis')
#######################################################################
def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1), np.arange(npt), np.sort(x))
#######################################################################

head=['VR', 'VT', 'VZ',  'Vtot', 'b', 'l', 'age',  'R', 'zb']
labell=[r"$V_{\rm{R}}$", r"$V_{\rm{T}}$", r"$V_{\rm{Z}}$", r"$V_{\rm{tot}}$", r"$b$", r"$l$", r"$Age$", r"$R$", r"$z$"]
df= pd.read_csv("./files/Gaia_filtered.txt", sep=" ",  skipinitialspace = True, header=None, usecols=[0,1,2,3,4,5,6,7],names=head)
print("describe:     ",  df.describe())
print("Columns:  ",  df.columns, "len(columns):  ",  len(df.columns))
print("******************************************************")

f1=open("./files/Gaia_filtered.txt","r")
nm= sum(1 for line in f1) 
par=np.zeros((nm,9)) 
par= np.loadtxt("./files/Gaia_filtered.txt")
par[:,4]=abs(par[:,4])

par[:,8]=abs(par[:,8]*1000.0)
############################################################

plt.clf()
fig, ax1= plt.subplots(figsize=(8, 6))
plt.scatter(par[:,6], par[:,8], marker= "^",facecolors='red', edgecolors='r', s= 3)
plt.xlabel(r"$Age$",fontsize=19,labelpad=0.0)
plt.ylabel(r"$Z$",fontsize=19,labelpad=0.2)
plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18, rotation=0)
#plt.ylim([0.02, 600])
#plt.xlim([2800, 11800])
#plt.yscale('log')
#plt.xscale('log')
plt.gca().invert_xaxis()
plt.grid("True")
plt.grid(linestyle='dashed')
fig= plt.gcf()
fig.savefig("./Histos/scater.jpg",dpi=200)
############################################################

corrM = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
corrM.style.background_gradient(cmap='coolwarm').set_precision(2)
ax= sns.heatmap(corrM, annot=True, xticklabels=labell, yticklabels=labell,annot_kws={"size": 16}, square=True, linewidth=1.0, cbar_kws={"shrink": .99}, linecolor="k",fmt=".3f", cbar=True, vmax=1, vmin=-1, center=0.0, ax=None, robust=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.xticks(rotation=45,horizontalalignment='right',fontweight='light', fontsize=18)
plt.yticks(rotation=0, horizontalalignment='right',fontweight='light', fontsize=18)
plt.title(r"$\rm{Correlation}~\rm{Matrix}$", fontsize=19)
fig.tight_layout()
plt.savefig("./Histos/corrMatrix.jpg", dpi=200)
print("**** Correlation matrix was calculated ******** ")

#######################################################################
for i in range(4):
    plt.cla()
    plt.clf()
    plt.subplots(figsize=(8,6))
    plt.plot( abs(par[:,i]), par[:,8] , "ro", markersize=2.0)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.xlabel(str(labell[i]),fontsize=17,labelpad=0.0)
    plt.ylabel(str(labell[4]),fontsize=17,labelpad=0.0)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    fig3= plt.gcf()
    fig3.savefig("./Histos/scatterVel1_{0:d}.jpg".format(i),dpi=200)
    
########################################################################
N=25;
ng=N-1
stds= np.zeros((ng*ng, 6))
iqrs= np.zeros((ng*ng, 6))

n0, bins0, patches = plt.hist( par[:,6], histedges_equalN(par[:,6],N) )
print("Age_Bins:    ",  np.min(par[:,6]), np.max(par[:,6]),  n0,   bins0)
age= bins0
arry=np.zeros(( int(np.max(n0)) , 9 ))                


count=0; 
for i in range(ng):
    l=0
    for j in range(nm): 
        if(float((par[j,6]-age[i])*(par[j,6]-age[i+1]))<0.0  or par[j,6]==age[i] ): 
            arry[l,:]=par[j,:]
            l+=1
    if(l!=int(n0[i])): 
        print("Error:  ",  l,  int(n0[i]) )
        input("Enter a number ")   
    n1, bins1, patches = plt.hist(arry[:l,8],histedges_equalN(arry[:l,8],N) ) 
    print("***************************************************************") 
    print("AGE:  ",  age[i],   n0[i] )   
    print("Height_bins:   ",np.min(par[:,8]), np.max(par[:,8]),  n1,   bins1)
    zz= bins1
    grid=np.zeros(( int(np.max(n1)) , 9))                
    ########################################
    for k in range(ng):
        m=0
        for j in range(l): 
            if(float((arry[j,8]-zz[k])*(arry[j,8]-zz[k+1]))<0.0 or arry[j,8]==zz[k] ): 
                grid[m,:]=arry[j,:]
                m+=1
        if(m != int(n1[k]) ): 
            print("Error:  ",  m,  int(n1[k]) )
            input("Enter a number ")      
        
        stds[count,:]=np.array([ np.mean(grid[:m,6]), np.mean(grid[:m,8]),np.std(grid[:m,0]), np.std(grid[:m,1]), 
            np.std(grid[:m,2]), np.std(grid[:m,3]) ])
        
        iq1= iqr(grid[:m,0],axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False) 
        iq2= iqr(grid[:m,1],axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False)   
        iq3= iqr(grid[:m,2],axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False)   
        iq4= iqr(grid[:m,3],axis=None, rng=(25, 75), scale=1.0, nan_policy='propagate', interpolation='linear', keepdims=False)   
        iqrs[count,:]=np.array([ np.mean(grid[:m,6]), np.mean(grid[:m,8]), iq1, iq2, iq3, iq4 ])
        count+=1 

fild=open("./files/stds.txt","w")
np.savetxt(fild,stds[:count,:].reshape(-1,6),fmt="%-9.5f    %-9.5lf       %9.5f    %9.5f     %9.5f     %9.5f ")
fild.close()
filf=open("./files/iqrs.txt","w")
np.savetxt(filf,iqrs[:count,:].reshape(-1,6),fmt="%-9.5f    %-9.5lf       %9.5f    %9.5f     %9.5f     %9.5f ")
filf.close()
##############################################################
head0=['age', 'z', 'VR',  'VT',  'VZ', 'Vtot' ]
lab=[r"$Age$", r"$z(\rm{pc})$", r"$V_{\rm{R}}(\rm{km/s})$", r"$V_{\rm{T}}(\rm{km/s})$", r"$V_{\rm{Z}}(\rm{km/s})$", r"$V_{\rm{tot}}(\rm{km/s})$"]
lab0=[r"$\rm{Age}$", r"$z$", r"$V_{\rm{R}}$", r"$V_{\rm{T}}$", r"$V_{\rm{Z}}$", r"$V_{\rm{tot}}$"]
df= pd.DataFrame(stds[:count,:], columns = head0)
corrM = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
corrM.style.background_gradient(cmap='coolwarm').set_precision(2)
ax= sns.heatmap(corrM, annot=True, xticklabels=lab0, yticklabels=lab0 , annot_kws={"size": 19}, square=True, 
linewidth=1.0, cbar_kws={"shrink": .99}, linecolor="k",fmt=".3f", cbar=True, vmax=1, vmin=0, center=0.0, ax=None, robust=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.xticks(rotation=0,horizontalalignment='right',fontweight='light', fontsize=18)
plt.yticks(rotation=0, horizontalalignment='right',fontweight='light', fontsize=18)
plt.title(r"$\rm{Correlation}~\rm{Matrix}$", fontsize=19)
fig.tight_layout()
plt.savefig("./Histos/corrMatrix_std.jpg", dpi=200)
print("**** Correlation matrix was calculated ******** ")
##############################################################

df= pd.DataFrame(iqrs[:count,:], columns = head0)
corrM = df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
corrM.style.background_gradient(cmap='coolwarm').set_precision(2)
ax= sns.heatmap(corrM, annot=True, xticklabels=lab0, yticklabels=lab0,annot_kws={"size": 19}, square=True, 
linewidth=1.0, cbar_kws={"shrink": .99}, linecolor="k",fmt=".3f", cbar=True, vmax=1, vmin=0, center=0.0, ax=None, robust=True)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=18)
plt.xticks(rotation=0,horizontalalignment='right',fontweight='light', fontsize=18)
plt.yticks(rotation=0, horizontalalignment='right',fontweight='light', fontsize=18)
plt.title(r"$\rm{Correlation}~\rm{Matrix}$", fontsize=19)
fig.tight_layout()
plt.savefig("./Histos/corrMatrix_iqr.jpg", dpi=200)
print("**** Correlation matrix was calculated ******** ")
##############################################################

for i in range(4):
    plt.cla()
    plt.clf()
    plt.subplots(figsize=(8,6))
    plt.plot( stds[:count,1] , stds[:count,2+i], "ro", markersize=4.0)
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(str(lab[1]),fontsize=17,labelpad=0.0)
    plt.ylabel(str(lab[2+i]),fontsize=17,labelpad=0.0)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    fig3= plt.gcf()
    fig3.savefig("./Histos/scatterVel2_{0:d}.jpg".format(i),dpi=200)
    
##############################################################  
y1=[29.0, 19.0, 12.0, 17.5]
y2=[59.0, 41.0, 28.0, 37.8]  
for i in range(4):
    plt.cla()
    plt.clf()
    plt.subplots(figsize=(8,6))
    plo= plt.scatter(iqrs[:count,1] , iqrs[:count,2+i],s=32.0,c=iqrs[:count,0], alpha=1.0)
    cbar = plt.colorbar(plo)
    cbar.set_label(r"$\rm{Age}(\rm{Gyr})$", fontsize=18.0)
    cbar.ax.tick_params(labelsize=16.)
    #plt.rcParams["figure.figsize"] = [7.00, 3.50]
    plt.rcParams["figure.autolayout"] = True
    plt.xticks(fontsize=17, rotation=0)
    plt.yticks(fontsize=17, rotation=0)
    plt.xlim([0.0,197.0])
    #plt.ylim([y1[i], y2[i]])
    #plt.yscale('log')
    #plt.xscale('log')
    plt.xlabel(str(lab[1]),fontsize=18,labelpad=0.0)
    plt.ylabel(str(lab[2+i]),fontsize=18,labelpad=0.0)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    fig3= plt.gcf()
    fig3.savefig("./Histos/scatterVel3_{0:d}.jpg".format(i),dpi=200)
    


















