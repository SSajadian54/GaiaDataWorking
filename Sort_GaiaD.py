import numpy as np
import matplotlib.pyplot as plt
import pylab as py
from matplotlib import rcParams
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import ( AutoLocator, AutoMinorLocator)
import pandas as pd
import math
####################################################################
year=float(365.2422)
parcs= 30856775814913673.0
au= 1.495978707*pow(10.0,11)
mm= au*0.001/(year*24.0*3600.0) 
Rsun=8.5;
####################################################################
def funcvel(v1, v2, v3, alf, delt, fla):

    vy= float(+np.cos(alf)*v1 +np.sin(alf)*(-np.sin(delt)*v2 +np.cos(delt)*v3) )
    vx= float(-np.sin(alf)*v1 +np.cos(alf)*(-np.sin(delt)*v2 +np.cos(delt)*v3) )
    vz= float(np.cos(delt)*v2+np.sin(delt)*v3) 
    VR= -0.0548755604162154*vx -0.8734370902348850 *vy -0.4838350155487132 *vz ### U
    VT= +0.4941094278755837*vx -0.4448296299600112 *vy +0.746982244497218  *vz ### V
    VZ= -0.8676661490190047*vx -0.1980763734312015 *vy +0.4559837761750669 *vz ### vertical  W
    #print "V1, v2, v3:  ",  v1,   v2,   v3 ,   np.sqrt(v1*v1 + v2*v2 + v3*v3 )
    #print "Vx, vy, vz:  ",  vx,   vy,   vz ,   np.sqrt(vx*vx + vy*vy + vz*vz )
    #print "VR, VT, VZ:  ",  VR,   VT,   VZ ,   np.sqrt(VR*VR + VT*VT + VZ*VZ )
    #print "*******************************************"
    if(fla==0):   
        VR +=11.1
        VT +=12.24
        VZ +=7.25
    if(fla==1):
        VR +=np.random.rand(1)*1.4-0.7
        VT +=np.random.rand(1)*1.0-0.5
        VZ +=np.random.rand(1)*0.8-0.4    
    return(VR, VT, VZ )
##########################################################################3 
#source_id,ra,dec,parallax,parallax_error,pmra,pmra_error,pmdec,pmdec_error,ruwe,visibility_periods_used,
#phot_g_mean_mag,phot_bp_mean_mag,phot_rp_mean_mag,phot_g_mean_flux_over_error,phot_bp_mean_flux_over_error,
#phot_rp_mean_flux_over_error,radial_velocity,radial_velocity_error,phot_variable_flag,l,b,classprob_dsc_combmod_star,
#teff_gspphot,logg_gspphot,mh_gspphot,ag_gspphot,mg_gspphot,mg_gspphot_lower,mg_gspphot_upper,alphafe_gspspec,radius_flame,
#lum_flame,mass_flame,age_flame,flags_flame,evolstage_flame


df = pd.read_csv("./Gaia_DR3t.csv")
nrow= len(df.source_id)
print ("n_row:  ",  nrow)

idd= df.source_id
unique, index = np.unique(idd, axis=0, return_index=True)
if(len(index) !=len(idd)): 
    print("They are different :  ",  len(index), len(idd))
    input("Enter a mnumber ")
    
data=np.zeros((nrow,9))
LT= np.zeros((nrow, 3))     
##########################################################################
k=0;  j=0; 
for i in range(nrow): 

    dis= abs(1.0/df.parallax[i])##kpc
    vra=  float(df.pmra[i]*mm*dis); ##km/s
    vde=  float(df.pmdec[i]*mm*dis);## km/s 
    Ervra=float(df.pmra_error[i]*mm*dis); ##km./s
    Ervde=float(df.pmdec_error[i]*mm*dis);##km/s 
    vr =  float(df.radial_velocity[i])## km/s
    Evr = float(df.radial_velocity_error[i])## km/s
    Right=float(df.ra[i]*np.pi/180.0)
    Dec=  float(df.dec[i]*np.pi/180.0)
    age=  float(df.age_flame[i])
    metal=float(df.mh_gspphot[i])
    mass= float(df.mass_flame[i])
    teff= float(df.teff_gspphot[i])
    lumi= float(df.lum_flame[i]) 
    flag= df.evolstage_flame[i]
    VR,VT,VZ=   funcvel(vra, vde , vr , Right , Dec, 0);   
    Ev1, Ev2, Ev3=funcvel(Ervra, Ervde , Evr ,Right ,Dec, 1);     
    vtot=np.sqrt(VR*VR+ VT*VT+ VZ*VZ)
    err1=float(df.phot_g_mean_flux_over_error[i])
    err2=float(df.phot_bp_mean_flux_over_error[i])
    err3=float(df.phot_rp_mean_flux_over_error[i])
    vis= float(df.visibility_periods_used[i])
    err4=float(df.parallax[i]/df.parallax_error[i])
    
    
    if(df.l[i]<=0.0): 
        df.l[i]=float(360.0+df.l[i])
    tet=float(360.0-df.l[i])*np.pi/180.0
    fi= abs(df.b[i]*np.pi/180.0)
    zb =abs(dis*np.sin(fi))
    yb =dis*np.cos(fi)*np.sin(tet);
    xb =Rsun-dis*np.cos(fi)*np.cos(tet);
    R=np.sqrt(xb*xb + yb*yb)
    
    if(teff>0.0  and lumi>0.0  and flag>0.0):  
        LT[j,0]= abs(teff)
        LT[j,1]= abs(lumi)
        LT[j,2]= abs(flag)
        j+=1

    if(err1>50.0 and err2>20.0 and err3>20.0 and err4>10.0 and vr>0.0 and float(df.ruwe[i])<1.5 and vis>7.0 and age>0.0): 
        data[k,  0]=float(VR)
        data[k,  1]=float(VT)
        data[k,  2]=float(VZ)
        data[k,  3]=abs(vtot)
        data[k,  4]=abs(df.b[i])
        data[k,  5]=float(df.l[i])
        data[k,  6]=abs(age)
        data[k,  7]=abs(R)
        data[k,  8]=abs(zb)
        k+=1 
    #else: 
    #    print(err1, err2, err3, err4,    Evr )
    #    print("ruwe,  vis:  ",  float(df.ruwe[i]),   vis)
    #    print( "Velocities:  ",  vtot, VR,   VT,  VZ, age)         
    if(i%10000==0): 
        print ("Step:  ", i, k)

  
fil1=open("Gaia_filtered.txt","w")
np.savetxt(fil1,data[:k,:].reshape(-1,9),fmt="%-9.5f    %-9.5lf    %-9.5f    %9.5f    %9.5f    %9.5f    %8.6f   %8.6f   %8.6f")
fil1.close()
fil2=open("cmd.txt","w")
np.savetxt(fil2,LT[:j,:].reshape(-1,3),fmt="%8.6f      %8.6f     %4.1f")
fil2.close()



############################################################ 
x1=[-200.0,-150.0,-150.0, 0.0,  0.0, 0.0 , 0.0,  8.2, 0.0]
x2=[200.0,150.0,150.0,  200.0 , 90.0,360.0,13.0, 8.8, 0.3 ]  
til1= [r"$V_{\rm R}$", r"$V_{\rm T}$", r"$V_{\rm Z}$", r"$V_{\rm{tot}}$", r"$\rm{latitude}$", r"$\rm{longtitude}$", r"$Age$", r"$R(kpc)$", r"$Z(kpc)$"] 
for i in range(9):        
    plt.clf()
    fig, ax1= plt.subplots(figsize=(8,6))
    plt.hist(data[:k,i],35, histtype='bar',ec='black',facecolor='green', alpha=0.8, rwidth=0.95)
    y_vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:.2f}'.format(x*1.0/k) for x in y_vals])
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.ylabel(r"$\rm{Distribution}$",fontsize=17,labelpad=0.0)
    plt.xlabel(str(til1[i]), fontsize=19)
    plt.xlim([ x1[i] , x2[i] ])
    plt.grid("True")
    plt.grid(linestyle='dashed')
    fig3= plt.gcf()
    fig3.savefig("./Histos/Histo1_{0:d}.jpg".format(i),dpi=200)
############################################################
til2=[r"$T_{\rm{eff}}(k)$", r"$Lumi(L_{\odot})$", r"$Evolutionary~Stage$"]
for i in range(3):        
    plt.clf()
    fig, ax1= plt.subplots(figsize=(8,6))
    plt.hist(LT[:j,i],35, histtype='bar',ec='black',facecolor='green', alpha=0.8, rwidth=0.95)
    y_vals = ax1.get_yticks()
    ax1.set_yticklabels(['{:.2f}'.format(x*1.0/k) for x in y_vals])
    plt.xticks(fontsize=16, rotation=0)
    plt.yticks(fontsize=16, rotation=0)
    plt.ylabel(r"$\rm{Distribution}$",fontsize=17,labelpad=0.0)
    plt.xlabel(str(til2[i]), fontsize=19)
    plt.grid("True")
    plt.grid(linestyle='dashed')
    fig3= plt.gcf()
    fig3.savefig("./Histos/Histo2_{0:d}.jpg".format(i),dpi=200)
############################################################

plt.clf()
fig, ax1= plt.subplots(figsize=(8, 6))
plt.scatter(LT[:j,0], LT[:j,1],marker= "^",facecolors='red', edgecolors='r', s= 14)
plt.ylabel(r"$\rm{L}_{\star}(L_{\odot})$",fontsize=19,labelpad=0.0)
plt.xlabel(r"$\rm{T}_{\rm{eff}}\rm{(K)}$",fontsize=19,labelpad=0.2)
plt.xticks(fontsize=18, rotation=0)
plt.yticks(fontsize=18, rotation=0)
#plt.ylim([0.02, 600])
#plt.xlim([2800, 11800])
plt.yscale('log')
plt.xscale('log')
plt.gca().invert_xaxis()
plt.grid("True")
plt.grid(linestyle='dashed')
fig= plt.gcf()
fig.savefig("./Histos/CMD.jpg",dpi=200)
############################################################


