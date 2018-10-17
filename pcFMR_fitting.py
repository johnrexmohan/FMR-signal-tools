from numpy import *
import pylab
import scipy

from scipy.optimize import curve_fit

import sys
sys.path.append(dir) # Specify your data directory here

from FMR_fitting_functions import *


# Specify the plotting preference and the frequency range
PLOT_all = False
Is_Plot = True
f = arange(3.0, 25.1, 1.0)


### FMR signal fitting
h0=[]
dh=[]
phi=[]
amp=[]
dh_err=[]
phi_err=[]

fig=pylab.figure()

for fi in f:
    fn="test\\"+str(fi)+"ghz.txt"
    h, sig = load_data(fn)

    p=initial_guess_1(h, sig)
    s = curve_fit(func_, h, sig, p)
    
    h0.append(abs(s[0][1]))
    dh.append(abs(s[0][2]))
    phi.append(-abs(s[0][3]))
    amp.append(-abs(s[0][0]))
    dh_err.append(sqrt(s[1][2][2]))
    phi_err.append(sqrt(s[1][3][3]))
    
    sig_fit = data_fit_1(h, s[0])
    
    if Is_Plot==True:
        plot_data(h, sig, sig_fit, s, fi, fn, 1)
    
    if PLOT_all==True:
        pylab.plot(h, sig, "bo")
        pylab.plot(h, sig_fit, "r")


### Fit gyromagnetic ratio geff and the kittle equation
l2=fit_kittel_geff(h0, f)
geff=l2[2]
l1=fit_linear(f, dh, geff)[0]

l2=fit_kittel(h0, f, geff)
l1=fit_linear(f, dh, geff)[0]

alpha_error=sqrt(fit_linear(f, dh, geff)[1][1][1])
alpha_cal=calculate_alpha_error(f, dh_err, geff)

gamma=27.99*geff/2.

xx=arange(0., 25.1, 0.1)
yy=func_linewidth(xx, l1[0], l1[1], geff)


### Save the results to txt
material="Py"

Output_data=open("D:\\Research\\pcFMR\\test\\"+material+'_data'+'.txt', 'w')
for i in range(len(f)):
    Output_data.write(str(f[i])+' '+str(dh[i])+' '+str(h0[i])+' '+str(phi[i])+' '+str(dh_err[i])+'\n')

Output_parameter=open("D:\\Research\\pcFMR\\test\\"+material+'_parameter'+'.txt', 'w')
Output_parameter.write('alpha= '+str(l1[1])+'\n')
Output_parameter.write('alpha_error= '+str(alpha_error)+'\n')
Output_parameter.write('dh0= '+str(l1[0])+'\n')
Output_parameter.write('4piMs= '+str(l2[0])+'\n')
Output_parameter.write('Hk= '+str(l2[1])+'\n')
Output_parameter.write('geff= '+str(geff)+'\n')


### Plot the results
if PLOT_all == True:
    pylab.xlabel("H (Oe)")
    pylab.ylabel("Signal (V)")
else:
    ax1=fig.add_subplot(131)
    ax2=fig.add_subplot(132)
    ax3=fig.add_subplot(133)



    ax1.set_xlabel("f(GHz)")
    ax1.set_ylabel("phase")
    ax1.set_xlim(0.,25.)
    ax1.set_ylim(-3.14,3.14)

    ax2.set_xlabel("f(GHz)")
    ax2.set_ylabel("dH(Oe)")
    ax2.set_xlim(0.,25.)
    ax2.set_ylim(0.,200.)

    ax3.set_xlabel("f(GHz)")
    ax3.set_ylabel("H0")
    ax3.set_xlim(0.,25.)
    ax3.set_ylim(0,2500)

    ax1.errorbar(f,phi, yerr=phi_err, fmt="bo", label=material)
    ax1.plot(f,phi,"bo", label=material)
    ax1.plot(f,phi,"b")
    
    ax2.errorbar(f,dh, yerr=dh_err, fmt="bo", label=material)
    ax2.plot(xx,yy, "r")
    ax2.text(8, 20, r"$\alpha="+str(l1[1])[:7]+"$\n$\Delta H_0="+str(l1[0])[:7]+"$ Oe")

    ax3.plot(f,h0, "bo", label=material)
    ax3.plot(f,h0, "b")
    ax3.text(13, 700, r"$4 \pi M_s="+str(l2[0])[:7]+"$ Oe\n$H_k="+str(l2[1])[:7]+"$ Oe\n$g_{eff}="+str(geff)[:7]+"$")

    ax1.legend(loc="upper center")
    ax2.legend(loc="upper center")
    ax3.legend(loc="upper center")

pylab.show()



