from numpy import *
import pylab
import scipy
from scipy.optimize import curve_fit

import sys
sys.path.append(dir) # Specify your data directory here

from FMR_fitting_functions import *


# Specify the plotting preference, the frequency range and the signal splits
Is_Plot = False
f=arange(3.0, 24.1, 1.0)

splits={
3.0: 100,
4.0: 180,
5.0: 250,
6.0: 350,
7.0: 450,
8.0: 600,
9.0: 750,
10.0: 900,
11.0: 1000,
12.0: 1200,
13.0: 1400,
14.0: 1600,
15.0: 1800,
16.0: 2000,
17.0: 2400,
18.0: 2500,
19.0: 2800,
20.0: 3000,
21.0: 3300,
22.0: 3500,
23.0: 3700,
24.0: 4000
}


### FMR 2-peak signal fitting
amp1=[]
h01=[]
dh1=[]
phi1=[]
dh1_err=[]
phi1_err=[]
amp2=[]
h02=[]
dh2=[]
phi2=[]
dh2_err=[]
phi2_err=[]

for fi in f:
    fn="test\\"+str(fi)+"ghz.txt"
    h, sig = load_data(fn)
    #sig=flatten(sig)

    p=initial_guess_2(h, sig, splits[fi])
        
    s = curve_fit(func_2peak, h, sig, p)
    amp1.append(s[0][0])
    h01.append(s[0][1])
    dh1.append(abs(s[0][2]))
    phi1.append(s[0][3])
    dh1_err.append(sqrt(s[1][2][2]))
    phi1_err.append(sqrt(s[1][3][3]))
    amp2.append(s[0][4])
    h02.append(s[0][5])
    dh2.append(abs(s[0][6]))
    phi2.append(s[0][7])
    dh2_err.append(sqrt(s[1][6][6]))
    phi2_err.append(sqrt(s[1][7][7]))
        
    sig_fit = data_fit_2(h, s[0])
    if Is_Plot==True:
            plot_data(h, sig, sig_fit, s, fi, fn, 1)
    

### Fit gyromagnetic ratio geff and the kittle equation
l2=fit_kittel_geff(h01, f)
geff1=l2[2]
l1=fit_linear(f, dh1, geff1)[0]

m2=fit_kittel_geff(h02, f)
geff2=m2[2]
m1=fit_linear(f, dh2, geff2)[0]

l2=fit_kittel(h01, f, geff1)
l1=fit_linear(f, dh1, geff1)[0]

geff2=2.1

m2=fit_kittel(h02, f, geff2)
m1=fit_linear(f, dh2, geff2)[0]

alpha1_error=sqrt(fit_linear(f, dh1, geff1)[1][1][1])
alpha2_error=sqrt(fit_linear(f, dh2, geff2)[1][1][1])

print(l1, l2)
print(m1, m2)

print('\n')


gamma1=27.99*geff1/2.
gamma2=27.99*geff2/2.

xx=arange(0., 25.1, 0.1)
yy1=func_linewidth(xx, l1[0], l1[1], geff1)
yy2=func_linewidth(xx, m1[0], m1[1], geff2)


### Save the results to txt
material1="Co"
material2="Py"

Output_data=open("D:\\Research\\pcFMR\\test\\"+material1+'_data'+'.txt', 'w')
for i in range(len(f)):
    Output_data.write(str(f[i])+' '+str(dh1[i])+' '+str(h01[i])+' '+str(phi1[i])+' '+str(amp1[i])+' '+str(dh1_err[i])+' '+str(phi1_err[i])+'\n')

Output_parameter=open("D:\\Research\\pcFMR\\test\\"+material1+'_parameter'+'.txt', 'w')
Output_parameter.write('alpha= '+str(l1[1])+'\n')
Output_parameter.write('alpha_error= '+str(alpha1_error)+'\n')
Output_parameter.write('dh0= '+str(l1[0])+'\n')
Output_parameter.write('4piMs= '+str(l2[0])+'\n')
Output_parameter.write('Hk= '+str(l2[1])+'\n')
Output_parameter.write('geff= '+str(geff1)+'\n')

Output_data=open("D:\\Research\\pcFMR\\test\\"+material2+'_data'+'.txt', 'w')
for i in range(len(f)):
    Output_data.write(str(f[i])+' '+str(dh2[i])+' '+str(h02[i])+' '+str(phi2[i])+' '+str(amp2[i])+' '+str(dh2_err[i])+' '+str(phi2_err[i])+'\n')

Output_parameter=open("D:\\Research\\pcFMR\\test\\"+material2+'_parameter'+'.txt', 'w')
Output_parameter.write('alpha= '+str(m1[1])+'\n')
Output_parameter.write('alpha_error= '+str(alpha2_error)+'\n')
Output_parameter.write('dh0= '+str(m1[0])+'\n')
Output_parameter.write('4piMs= '+str(m2[0])+'\n')
Output_parameter.write('Hk= '+str(m2[1])+'\n')
Output_parameter.write('geff= '+str(geff2)+'\n')


### Plot the results
fig=pylab.figure()

ax1=fig.add_subplot(131)
ax2=fig.add_subplot(132)
ax3=fig.add_subplot(133)

ax1.set_xlabel("f(GHz)")
ax1.set_ylabel("phase")
ax1.set_ylim(-3.14,3.14)

ax2.set_xlabel("f(GHz)")
ax2.set_ylabel("dH")
ax2.set_xlim(0,25)
ax2.set_ylim(0,200)

ax3.set_xlabel("f(GHz)")
ax3.set_ylabel("H0")
ax3.set_xlim(0,30)
ax3.set_ylim(0,15000)

ax1.errorbar(f,phi2, yerr=phi2_err, fmt="bo", label=material2)
ax1.plot(f,phi2,"b")
ax1.errorbar(f,phi1, yerr=phi1_err, fmt="ro", label=material1)
ax1.plot(f,phi1,"r")

ax2.errorbar(f,dh2, yerr=dh2_err, fmt="bo", label=material2)
ax2.plot(xx,yy2, "b")
ax2.errorbar(f,dh1, yerr=dh1_err, fmt="ro", label=material1)
ax2.plot(xx,yy1, "r")
ax2.text(15, 40, r"Co20: $\alpha="+str(l1[1])[:7]+"$\n$\Delta H_0="+str(l1[0])[:7]+"$"+\
"\n"+r"Py12: $\alpha="+str(m1[1])[:7]+"$\n$\Delta H_0="+str(m1[0])[:7]+"$")

ax3.plot(f,h02, "bo", label=material2)
ax3.plot(f,h02, "b")
ax3.plot(f,h01, "ro", label=material1)
ax3.plot(f,h01, "r")
ax3.text(12, 400, r"Co20: $4 \pi M_s="+str(l2[0])[:7]+"$ Oe\n$H_k="+str(l2[1])[:7]+"$ Oe\n$g_{eff}="+str(geff1)[:7]+"$"+\
"\n"+r"Py12: $4 \pi M_s="+str(m2[0])[:7]+"$ Oe\n$H_k="+str(m2[1])[:7]+"$ Oe\n$g_{eff}="+str(geff2)[:7]+"$")

ax1.legend(loc="upper center")
ax2.legend(loc="upper center")
ax3.legend(loc="upper center")

pylab.show()


