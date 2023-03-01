import numpy as np 
import pylab as py 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
from matplotlib import rcParams
import time
import matplotlib
rcParams["font.size"] = 11.5
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Computer Modern Sans"]
rcParams["text.usetex"] = True
rcParams["text.latex.preamble"] = r"\usepackage{cmbright}"
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.ticker import StrMethodFormatter
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn import linear_model,ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.tree import export_text
from sklearn.metrics import r2_score
from sklearn import tree
cmap=plt.get_cmap('viridis')

f1=open("./files/ntree.txt","r")
nm= sum(1 for line in f1) 
fore=np.zeros(( 500, 5)) 
fore= np.loadtxt("./files/ntree.txt")
#######################################################################
plt.clf()
fig= plt.figure(figsize=(8,6)) 
plt.plot(fore[:,0], fore[:,1], 'b--',lw=2.5, label=r"$v_{\rm{R}}$")
plt.plot(fore[:,0], fore[:,2], 'm--',lw=2.5, label=r"$v_{\rm{T}}$")
plt.plot(fore[:,0], fore[:,3], 'g--',lw=2.5, label=r"$v_{\rm{Z}}$")
plt.plot(fore[:,0], fore[:,4],'r--',lw=2.5, label=r"$v_{\rm{tot}}$")
plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18, rotation=0)
plt.xlabel(r"$\rm{Number}~\rm{of}~\rm{Trees}$",fontsize=18)
plt.ylabel(r"$\rm{Mean}~\rm{Absolute}~\rm{Percentage}~\rm{Error}[\%]$",fontsize=18)
plt.xlim([1,500])
plt.xscale('log')
#plt.ylim([5.0,10.0])
plt.legend()
plt.legend(loc='best',fancybox=True, shadow=True)
plt.legend(prop={"size":18})
plt.grid(True)
plt.grid(linestyle='dashed')
fig=plt.gcf()
fig.tight_layout()
fig.savefig("./Histos/notree.jpg", dpi=200)


#######################################################################

