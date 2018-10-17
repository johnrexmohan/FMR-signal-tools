import numpy as np
import scipy
from scipy.optimize import curve_fit
import pylab

geff=2.1
Is_Plot = False

def sign(x):
    """
    return 1 if >0
    return -1 if <0
    """
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
def func_(x, amp, x0, FWHM, phs, off):
    """
    Derivative Lorentzian shape
    """
    return amp*FWHM**2/4.*np.imag(np.exp(-1j*phs)/(x-x0+1j*FWHM/2.)**2) \
            +off

def func_Lor(x, amp, x0, FWHM, phs, off):
    """
    Lorentzian shape
    """
    return amp*np.imag(np.exp(-1j*phs)/(x-x0+1j*FWHM/2.)) \
            +off


def func_2peak(x, amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, off):

    return amp1*FWHM1**2/4.*np.imag(np.exp(-1j*phs1)/(x-x01+1j*FWHM1/2.)**2) \
            +amp2*FWHM2**2/4.*np.imag(np.exp(-1j*phs2)/(x-x02+1j*FWHM2/2.)**2) \
            +off

def func_2peak_fix_spacing(delta_H0):

    return lambda x, amp1, FWHM1, phs1, amp2, x02, FWHM2, phs2, off: \
            amp1*FWHM1**2/4.*np.imag(np.exp(-1j*phs1)/(x-(x02-delta_H0)+1j*FWHM1/2.)**2) \
            +amp2*FWHM2**2/4.*np.imag(np.exp(-1j*phs2)/(x-x02+1j*FWHM2/2.)**2) \
            +off

def func_3peak(x, amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, amp3, x03, FWHM3, phs3, off):

    return amp1*FWHM1**2/4.*np.imag(np.exp(-1j*phs1)/(x-x01+1j*FWHM1/2.)**2) \
            +amp2*FWHM2**2/4.*np.imag(np.exp(-1j*phs2)/(x-x02+1j*FWHM2/2.)**2) \
            +amp3*FWHM3**2/4.*np.imag(np.exp(-1j*phs3)/(x-x03+1j*FWHM3/2.)**2) \
            +off

def func_slope(x, amp, x0, FWHM, phs, slp, off):
    
    return amp*FWHM**2/4.*np.imag(np.exp(-1j*phs)/(x-x0+1j*FWHM/2.)**2)+x*slp \
            +off

def func_quad(x, amp, x0, FWHM, phs, slp, quad, off):
    
    return amp*FWHM**2/4.*np.imag(np.exp(-1j*phs)/(x-x0+1j*FWHM/2.)**2) \
            + x * slp + x**2 * quad \
            + off

def func_kittel(h0, ms, hk, geff):
    f = np.sqrt(((1.3995*geff)**2)*(h0+hk)*(h0+hk+ms)*1.e-6)
    return f

def func_kittel_out(f, ms):
    global geff
    h0 = ms + f*1000./1.3995/geff
    return h0

def func_kittel_out1(f, ms, geff):
    h0 = ms + f*1000./1.3995/geff
    return h0

def func_kittel_geff(h0, ms, hk, geff):
    #global geff
    f = np.sqrt(((1.3995*geff)**2)*(h0+hk)*(h0+hk+ms)*1.e-6)
    return f

def func_linewidth(f, dh0, alpha, geff):
    """
    For both in-plane and out-of-plane, Wei's
    """
    gamma=27.99*geff/2.
    dh = dh0+2.*alpha*f/gamma*10000.
    return dh

def func_linewidth_dh0(dh0):
    """
    For both in-plane and out-of-plane
    """
    global geff
    
    return lambda f, alpha: dh0 + alpha/0.700*f/geff*1e3

def func_linewidth_0(f, alpha):
    """
    For both in-plane and out-of-plane
    """
    global geff
    dh = alpha/0.700*f/geff*1e3
    return dh

def load_data(fn):
    h, sig = np.loadtxt(fn, usecols = (0,1), unpack = True)
    return h, sig

def cut_data(h, sig, hrange):
    h = np.array(h)
    sig = np.array(sig)
    h1, h2 = hrange
    N=len(h)
    n1=0
    n2=N-1
    
    for i in range(N):
        if h[i] > h1:
            n1 = i
            break

    for i in range(N):
        if h[i] > h2:
            n2 = i
            break
    h_cut=h[n1:n2+1]
    sig_cut=sig[n1:n2+1]
    
    return h_cut, sig_cut

def flatten(sig):
    """
    Remove the slope background
    """

    N=len(sig)
    slope=(sig[N-1]-sig[0])/(N-1)
    for i in range(N):
        sig[i]=sig[i]-slope*i

    return sig

def find_nearest(array,value):
    idx=(np.abs(array-value)).argmin()
    return idx

def separate_data(h, sig, split):
    idx = find_nearest(h, split)
    n = len(h)
    h1 = h[0:idx]
    h2 = h[idx:n]    
    sig1 = sig[0:idx]
    sig2 = sig[idx:n]
    
    return h1, sig1, h2, sig2

def get_H02(f, path):
    fn=path+"\\"+str(f)+"ghz.txt"
    h, sig = load_data(fn)
    sig=flatten(sig)
    p=initial_guess_1(h, sig)
    s = curve_fit(func_, h, sig, p)
    return s[0][1]

def initial_guess_1(h, sig):
    h=np.array(h)
    sig=np.array(sig)
    
    i1 = sig.argmax()
    i2 = sig.argmin()
    amp  = (sig.max() - sig.min())/2 * 1.7
    x0 = (h[i1] + h[i2])/2
    FWHM = (h[i1] - h[i2]) * 1.73
    phs=0
    off = 0
    
    p = np.array([amp, x0, FWHM, phs, off])
    return p

def initial_guess_1_slp(h, sig):
    h=np.array(h)
    sig=np.array(sig)
    
    i1 = sig.argmax()
    i2 = sig.argmin()
    amp  = (sig.max() - sig.min())/2 * 1.7
    x0 = (h[i1] + h[i2])/2
    FWHM = (h[i1] - h[i2]) * 1.73
    phs = 0
    slp = 0
    off = 0
    
    p = np.array([amp, x0, FWHM, phs, slp, off])
    return p

def initial_guess_1_quad(h, sig):
    h=np.array(h)
    sig=np.array(sig)
    
    i1 = sig.argmax()
    i2 = sig.argmin()
    amp  = (sig.max() - sig.min())/2 * 1.7
    x0 = (h[i1] + h[i2])/2
    FWHM = (h[i1] - h[i2]) * 1.73
    phs = 0
    slp = 0
    quad = 0
    off = 0
    
    p = np.array([amp, x0, FWHM, phs, slp, quad, off])
    return p

def initial_guess_2_slope(h, sig):
    n = len(h)
    slope = (sig[n-1] - sig[0]) / (h[n-1] - h[0])
    
    x = np.linspace(0, 1.0, n) * (sig[n-1] - sig[0])
    sig_ = sig - x

    i1 = sig_.argmax()
    i2 = sig_.argmin()
    amp  = (sig_.max() - sig_.min())/2 * 1.7
    x0 = (h[i1] + h[i2])/2
    FWHM = (h[i1] - h[i2]) * 1.73
    phs=0
    
    off = 0
    
    p = np.array([amp, x0, FWHM, phs, slope, off])
    return p 


def initial_guess_2(h, sig, arg):
    #arg=[split,i]
    #i=1 -> main peak left
    #i=2 -> main peak right
    split=arg
    #idx_main=arg[1]
    
    n = len(h)
    for i in range(n):
        if h[i] > split:
            break
    h1 = h[0:i]
    h2 = h[i:n]
    sig1 = np.array(sig[0:i])
    sig2 = np.array(sig[i:n])
    
    p1 = initial_guess_1(h1,sig1)
    p2 = initial_guess_1(h2,sig2)

    return list(p1[0:4])+list(p2)

def initial_guess_3(h, sig, arg):
    #arg=[split,i]
    #i=1 -> main peak left
    #i=2 -> main peak right
    split=arg
    #idx_main=arg[1]
    
    n = len(h)
    for i in range(n):
        if h[i] > split[0]:
            break
    a=i
    for i in range(n):
        if h[i] > split[1]:
            break    
    b=i
    h1 = h[0:a]
    h2 = h[a:b]
    h3 = h[b:n]
    sig1 = np.array(sig[0:a])
    sig2 = np.array(sig[a:b])
    sig3 = np.array(sig[b:n])
    
    p1 = initial_guess_1(h1,sig1)
    p2 = initial_guess_1(h2,sig2)
    p3 = initial_guess_1(h3,sig3)

    return list(p1[0:4])+list(p2[0:4])+list(p3)

def initial_guess_22(h, sig, arg):
    #arg=[split,i]
    #i=1 -> main peak left
    #i=2 -> main peak right
    split=arg[0]
    #idx_main=arg[1]
    
    n = len(h)
    for i in range(n):
        if h[i] > split:
            break
    h1 = h[0:i]
    h2 = h[i:n]
    sig1 = np.array(sig[0:i])
    sig2 = np.array(sig[i:n])
    
    #if idx_main == 1:
    p1 = initial_guess_1(h1,sig1)
    p2 = initial_guess_1(h2,sig2)
    
    
    return list(p1[0:4])+list(p2)

def fit_Lorenzian_1(h, sig, p):
    s = curve_fit(func_, h, sig, p)
    #print "\n\nfitted parameters:"
    #print "amp, x0, FWHM, phs, off:"
    #print s[0]
    return s[0]
    
def fit_Lorenzian_2(h, sig, p):
    s = curve_fit(func_2peak, h, sig, p)
    #print "\n\nfitted parameters:"
    #print "amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, off:"
    #print s[0]
    return s[0]

def fit_Lorenzian_3(h, sig, p):
    s = curve_fit(func_3peak, h, sig, p)
    #print "\n\nfitted parameters:"
    #print "amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, off:"
    #print s[0]
    return s[0]

def fit_SWR_2peak(h, sig, delta_H0):
    # 1. trial fit
    p = initial_guess_1(h, sig)
    s = fit_Lorenzian_1(h, sig, p)
    
    off = s[4]
    s1=[s[0], s[2], s[3]]
    s2=list(s)[0:4]
    p_2 = s1+s2+[off]

    # obtain trial fitting parameters
    p=curve_fit(func_2peak_fix_spacing(delta_H0), h, sig, p_2)[0]
    #print p
    
    # 2. real fit
    
    pp = [p[0], p[4]-delta_H0, p[1], p[2], p[3], p[4], p[5], p[6], p[7]]
    
    ss = fit_Lorenzian_2(h, sig, pp)
    #print ss
    
    return ss

def fit_SWR_3peak(h, sig, delta_H01, delta_H02, hrange_01):
    """
    delta_H01 -> 1st SWR
    delta_H02 -> 2nd SWR
    use hrange_01 to pre-fit uniform mode and 1st SWR
    """
    # 1. trial fit
    h_cut, sig_cut = cut_data(h, sig, hrange_01)
    p = initial_guess_1(h, sig)
    s = fit_Lorenzian_1(h, sig, p)
    
    off = s[4]
    s1=[s[0]/5.0, s[2], s[3]]
    s2=list(s)[0:4]
    p_2 = s1+s2+[off]

    # obtain trial fitting parameters
    p1=curve_fit(func_2peak_fix_spacing(delta_H01), h_cut, sig_cut, p_2)[0]
    p2=curve_fit(func_2peak_fix_spacing(delta_H02), h, sig, p_2)[0]
    
    #p1 = fit_SWR_2peak(h_cut, sig_cut, delta_H01)
    #p2 = fit_SWR_2peak(h, sig, delta_H02)
    
    # 2. real fit
    
    p_try1 = [p1[0], p1[4]-delta_H01, p1[1], p1[2]]
    p_try2 = [p2[0], p2[4]-delta_H02, p2[1], p2[2]]
    
    p_try = p_try1+p_try2+[p1[3], p1[4], p1[5], p1[6], p1[7]]
    
    ss = fit_Lorenzian_3(h, sig, p_try)
    #print ss
    
    return ss

def fit_Lorenzian_slope(h, sig, p):
    s = curve_fit(func_slope, h, sig, p)
    #print "\n\nfitted parameters:"
    #print "amp, x0, FWHM, phs, slope, off:"
    #print s[0]
    return s[0]

def data_fit_1(h, p):
    amp=p[0];
    x0=p[1];
    FWHM=p[2];
    phs=p[3];
    off=p[4];
    
    sig=func_(h, amp, x0, FWHM, phs, off)
    return sig

def data_fit_1_slp(h, p):
    amp=p[0]
    x0=p[1]
    FWHM=p[2]
    phs=p[3]
    slp=p[4]
    off=p[5]
    
    sig=func_slope(h, amp, x0, FWHM, phs, slp, off)
    return sig

def data_fit_1_quad(h, p):
    amp=p[0]
    x0=p[1]
    FWHM=p[2]
    phs=p[3]
    slp=p[4]
    quad=p[5]
    off=p[6]
    
    sig=func_quad(h, amp, x0, FWHM, phs, slp, quad, off)
    return sig

def data_fit_2(h, p):
    amp1=p[0];
    x01=p[1];
    FWHM1=p[2];
    phs1=p[3];
    amp2=p[4];
    x02=p[5];
    FWHM2=p[6];
    phs2=p[7];
    off=p[8];
    
    sig1=func_2peak(h, amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, off)
    
    return sig1
    
def data_fit_3(h, p):
    amp1=p[0];
    x01=p[1];
    FWHM1=p[2];
    phs1=p[3];
    amp2=p[4];
    x02=p[5];
    FWHM2=p[6];
    phs2=p[7];
    amp3=p[8];
    x03=p[9];
    FWHM3=p[10];
    phs3=p[11];
    off=p[12];
    
    sig=func_3peak(h, amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, amp3, x03, FWHM3, phs3, off)
    return sig
def data_fit_slope(h, p):
    amp=p[0];
    x0=p[1];
    FWHM=p[2];
    phs=p[3];
    slope=p[4];
    off=p[5];
    
    sig1=func_slope(h, amp, x0, FWHM, phs, slope, off)
    
    return sig1

def plot_data(h, sig, sig_fit, p, f, fn, arg):
    """
    p -> fitting result
    f -> frequency, used in the title
    fn -> file name, used to save
    arg -> 0: show
           1: savefig
           2: pass
    """
    x_lim = [h.min(), h.max()]
    y_lim = [sig.min(), sig.max()]
    at_x = 0.03 * x_lim[1] + 0.97 * x_lim[0]   #x position of annotation
    at_y = 0.9 * y_lim[1] + 0.1 * y_lim[0]   #y position of annotation
   
    
    pylab.figure()
    pylab.plot(h, sig, 'bo')
    pylab.plot(h, sig_fit, 'r')
    pylab.xlim(x_lim)
    #pylab.y_lim(x_lim)
    pylab.title(str(f)+"GHz")
    pylab.xlabel('H (Oe)')
    pylab.ylabel('Signal (V)')
    #pylab.text(at_x, at_y, 'Ho = '+str(p[1])[:6]+'Oe\n'+'dH = '+str(p[2])[:6]+'Oe\n'+'phs = '+str(p[4])[:4], fontsize = 14)
    if arg == 0:
        pylab.show()
    elif arg == 1:
        pylab.savefig(fn.replace('.txt','(Yi).png'))
        pylab.close()
    elif arg == 2:
        pass

def plot_data_2(h1, sig1, sig_fit1, h2, sig2, sig_fit2, f, fn):
    x_lim = [min(h1.min(),h2.min()), max(h1.max(),h2.max())]
    y_lim = [min(sig1.min(),sig2.min()), max(sig1.max(),sig2.max())]
    at_x = 0.03 * x_lim[1] + 0.97 * x_lim[0]   #x position of annotation
    at_y = 0.9 * y_lim[1] + 0.1 * y_lim[0]   #y position of annotation
   
    
    pylab.figure()
    pylab.plot(h1, sig1, 'bo')
    pylab.plot(h2, sig2, 'bo')
    pylab.plot(h1, sig_fit1, 'r')
    pylab.plot(h2, sig_fit2, 'r')
    pylab.xlim(x_lim)
    #pylab.y_lim(x_lim)
    pylab.title(str(f)+"GHz")
    pylab.xlabel('H (Oe)')
    pylab.ylabel('Signal (V)')
    #pylab.text(at_x, at_y, 'Ho = '+str(p[1])[:6]+'Oe\n'+'dH = '+str(p[2])[:6]+'Oe\n'+'phs = '+str(p[4])[:4], fontsize = 14)
    #pylab.show()
    pylab.savefig(fn.replace('.txt','(Yi).png'))
    pylab.close()

def Fit_Lor_1(fn, f):
    """
    Single layer structure with 1 resonances
    """
    
    global Is_Plot
    
    h, sig = load_data(fn)
    sig=flatten(sig)
    p = initial_guess_1(h, sig)
    s = fit_Lorenzian_1(h, sig, p)
    sig_fit = data_fit_1(h, s)
    
    ##plot individual fitting result
    if Is_Plot == True:
        plot_data(h, sig, sig_fit, s, f, fn, 1)

    ##amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, off:
    ph1=calculate_phs(s[0], s[2], s[3])

    ##s = [amp, x0, FWHM, phs, off]
    return s[1], abs(s[2]), ph1

def Fit_Lor_2(fn, f, split):
    """
    Trilayer structure with 2 resonances
    Use split to find resonance field and run "initial_guess_2" function
    """
    
    global Is_Plot
    
    h, sig = load_data(fn)
    p = initial_guess_2(h, sig, split)
    s = fit_Lorenzian_2(h, sig, p)
    sig_fit = data_fit_2(h, s)
    
    print s
    
    ##plot individual fitting result
    if Is_Plot == True:
        plot_data(h, sig, sig_fit, s, f, fn, 1)
        
    ##amp1, x01, FWHM1, phs1, amp2, x02, FWHM2, phs2, off:
    ph1=calculate_phs(s[0], s[2], s[3])
    ph2=calculate_phs(s[4], s[6], s[7])

    ##    return H1, H2, d_H1, d_H2, ph1, ph2
    return s[1], s[5], abs(s[2]), abs(s[6]), ph1, ph2
    
def Fit_Lor_2_separate(fn, f, split):
    """
    Trilayer structure with 2 resonances
    Use split to find resonance field and run "initial_guess_2_slope" function
    """
    
    global Is_Plot
    
    h, sig = load_data(fn)
    h1, sig1, h2, sig2 = separate_data(h, sig, split)
    
    p1 = initial_guess_2_slope(h1, sig1)
    p2 = initial_guess_2_slope(h2, sig2)
    s1 = fit_Lorenzian_slope(h1, sig1, p1)
    s2 = fit_Lorenzian_slope(h2, sig2, p2)
    sig_fit1 = data_fit_slope(h1, s1)
    sig_fit2 = data_fit_slope(h2, s2)
    
    ##plot individual fitting result
    if Is_Plot == True:
        plot_data_2(h1, sig1, sig_fit1, h2, sig2, sig_fit2, f, fn)

    ##s = [amp, x0, FWHM, phs, off]
    a1=(1-sign(s1[0]))*np.pi/2
    a2=(1-sign(s2[0]))*np.pi/2
    return s1[1], s2[1], s1[2], s2[2], s1[3]+a1, s2[3]+a2



def Fit_SWR_X(f, path_SWR, path_uniform, hranges_guess, hranges_fit):
    """
    Fit SWR signal, with the help of the main resonance peak
    1. go to path_uniform and find the H0 of the uniform mode
    2. take 1/(h-H0)^2 as the background fitting function
    3. use hranges_guess to have an initial guess
    4. use func_X to fit the SWR, in the range of hranges_fit
    5. return h0, dh, phi
    """
    h02=get_H02(f, path_uniform)
    
    fn=path_SWR+"\\"+str(f)+"ghz.txt"
    h, sig = load_data(fn)
    h_cut, sig_cut = load_data(fn)
    
    if hranges_guess.keys().count(f)>0:
        hrange = hranges_guess[f]
        h_cut, sig_cut = cut_data(h, sig_cut, hrange)
    if hranges_fit.keys().count(f)>0:
        hrange = hranges_fit[f]
        h, sig = cut_data(h, sig, hrange)
    sig_cut=flatten(sig_cut)
    
    p=initial_guess_1(h_cut, sig_cut)
    amp2_guess=sig[-1]-sig[0]
    p_new = list(p)+[amp2_guess]
    
    s = curve_fit(func_X(h02), h, sig, p_new)
    print s[0]
    
    if Is_Plot == True:
            sig_fit=func_X(h02)(h, s[0][0], s[0][1], s[0][2], s[0][3], s[0][4], s[0][5])
            pylab.figure()
            pylab.plot(h, sig, "bo")
            pylab.plot(h, sig_fit, "r")
            pylab.savefig(path_SWR+"\\"+str(f)+"ghz(Yi_compreh).png")
            pylab.close()

    amp_temp=s[0][0]
    h0_temp=s[0][1]
    dh_temp=s[0][2]
    ph_temp=s[0][3]
    ph_temp=calculate_phs(amp_temp, dh_temp, ph_temp)
        
    return h0_temp, abs(dh_temp), ph_temp

def Fit_SWR_X_shift(f, path_SWR, H_shift, hranges_guess, hranges_fit):
    """
    Fit SWR signal, with the help of the main resonance peak
    1. use hranges_guess to have an initial guess
    2. use the given shift to calculate H0
    3. take 1/(h-H0)^2 as the background fitting function
    4. use func_X to fit the SWR, in the range of hranges_fit
    5. return h0, dh, phi
    """
    fn=path_SWR+"\\"+str(f)+"ghz.txt"
    h, sig = load_data(fn)
    h_cut, sig_cut = load_data(fn)
    
    if hranges_guess.keys().count(f)>0:
        hrange = hranges_guess[f]
        h_cut, sig_cut = cut_data(h, sig_cut, hrange)
    if hranges_fit.keys().count(f)>0:
        hrange = hranges_fit[f]
        h, sig = cut_data(h, sig, hrange)
    sig_cut=flatten(sig_cut)
    
    p=initial_guess_1(h_cut, sig_cut)
    h02 = H_shift + p[1]
    amp2_guess=p[0]
    p_new = list(p)+[amp2_guess]  
    
    s = curve_fit(func_X(h02), h, sig, p_new)
    print s[0]
    
    if Is_Plot == True:
            sig_fit=func_X(h02)(h, s[0][0], s[0][1], s[0][2], s[0][3], s[0][4], s[0][5])
            pylab.figure()
            pylab.plot(h, sig, "bo")
            pylab.plot(h, sig_fit, "r")
            pylab.savefig(path_SWR+"\\"+str(f)+"ghz(Yi_compreh).png")
            pylab.close()
    
    h0, dh, phi = s[0][1], abs(s[0][2]), regulate_phs(s[0][3])
    return h0, dh, phi

def fit_linear(f,dH, geff):
    """
    For both in-plane and out-of-plane
    """
    f = np.array(f)
    dH = np.array(dH)
    
    p = curve_fit(lambda f, dh0, alpha: func_linewidth(f, dh0, alpha, geff), f, dH, (0. ,0.007))
    #print p
    return p

def fit_linear_dh0(f, dH, dh0):
    """
    For both in-plane and out-of-plane
    """
    f = np.array(f)
    dH = np.array(dH)
    
    p = curve_fit(func_linewidth_dh0(dh0), f, dH, (0.,))
    print p
    return p[0]


def fit_linear_0(f,dH):
    """
    For both in-plane and out-of-plane
    """
    f = np.array(f)
    dH = np.array(dH)
    
    p = curve_fit(func_linewidth_0, f, dH, (0.007,))
    print p
    return p[0]

def fit_kittel(H0, f, geff):
    f = np.array(f)
    H0 = np.array(H0)
    s = curve_fit(lambda h0, ms, hk: func_kittel(h0, ms, hk, geff), H0, f, (10000., 1.))
    #print s
    return s[0]

def fit_kittel_out(f,H0):
    f = np.array(f)
    H0 = np.array(H0)
    s = curve_fit(func_kittel_out, f, H0, (10000,))
    print s
    return s[0]

def fit_kittel_out1(f,H0):
    f = np.array(f)
    H0 = np.array(H0)
    s = curve_fit(func_kittel_out1, f, H0, (10000, 2.0))
    print s
    return s[0]

def fit_kittel_geff(H0,f):
    #print f
    #print H0
    f = np.array(f)
    H0 = np.array(H0)
    s = curve_fit(func_kittel_geff, H0, f, (10000., 1., 2.))
    print s
    return s[0]

def regulate_phs(phs):
    """
    Regulate the phase to (-pi, pi)
    """
    phs1=phs
    pi=np.pi
    while phs1 > 2*pi:
        phs1 =phs1-2*pi
    while phs1 < 0*pi:
        phs1 =phs1+2*pi
    
    return phs1

def calculate_phs(amp, dh, phs):
    """
    1. If dh < 0, reverse amp, dh and phs
    2. If amp <0, add pi phase on phs
    3. regulate phs in the range
    """
    if dh<0:
        amp1=-amp
        dh1=-dh
        phs1=-phs
    else:
        amp1=amp
        dh1=dh
        phs1=phs
    
    phs1=phs1+(1-sign(amp1))*np.pi/2
    
    phs1=regulate_phs(phs1)
    
    return phs1
    
def calculate_alpha_error(f, dh_err, geff):
    f=np.array(f)
    dh_err=np.array(dh_err)
    avg=sum(f)/len(f)
    tol=0.
    for i in range(len(f)):
        tol=tol+(f[i]-avg)**2
    fenmu=tol
    tol=0.
    for i in range(len(f)):
        tol=tol+((f[i]-avg)**2)*(dh_err[i]**2)
    fenzi=np.sqrt(tol)
    factor=27.99*geff/40000.
    return(fenzi/fenmu*factor)

def main_fit_1(file, freqs):
    H = []
    d_H = []
    phs_ = []
    n = len(freqs)
    for i in range(n):
        path = file+'\\'+str(freqs[i])+"ghz.txt"
        h, dH, phs, = Fit_Lor_1(path, freqs[i])
        #phs1 = regulate_phs(phs1)
        H.append(h)
        d_H.append(dH)
        phs_.append(phs)
        
    return H, d_H, phs_

def main_fit_2(file, freqs, splits):
    H1 = []
    H2 = []
    d_H1 = []
    d_H2 = []
    ph1 = []
    ph2 = []
    n = len(freqs)
    for i in range(n):
        path = file+'\\'+str(freqs[i])+"ghz.txt"
        h1, h2, dH1, dH2, phs1, phs2 = Fit_Lor_2(path, freqs[i], splits[freqs[i]])

        H1.append(h1)
        H2.append(h2)
        d_H1.append(dH1)
        d_H2.append(dH2)
        ph1.append(phs1)
        ph2.append(phs2)
    
    return np.array(H1), np.array(H2), np.array(d_H1), np.array(d_H2), np.array(ph1), np.array(ph2)


def main_fit_2_sep(file, freqs, splits):
    H_1 = []
    H_2 = []
    d_H1 = []
    d_H2 = []
    ph1 = []
    ph2 = []
    n = len(freqs)
    for i in range(n):
        #print "FFFF=",i
        path = file+'\\'+str(freqs[i])+".0ghz.txt"
        H1, H2, dH1, dH2, phs1, phs2 = Fit_Lor_2_separate(path, freqs[i], splits[i])
        phs1 = regulate_phs(phs1)
        phs2 = regulate_phs(phs2)
        H_1.append(H1)
        H_2.append(H2)
        d_H1.append(dH1)
        d_H2.append(dH2)
        ph1.append(phs1)
        ph2.append(phs2)
    
    return H_1, H_2, d_H1, d_H2, ph1, ph2

def plot_phs(freqs, ph, ax1, mark, lbl):
    ax1.plot(freqs,ph,mark[0],label=lbl)
    ax1.plot(freqs,ph,mark[1])

def plot_linewidth(freqs,d_H, ax2, mark, lbl):
    ax2.plot(freqs,d_H,mark[0],label=lbl)
def plot_kittel(freqs, H, ax, mark, lbl):
    ax.plot(freqs, H, mark[0], label=lbl)
    ax.plot(freqs, H, mark[1])
def add_plottxt(ax, text, rx, ry):
    x1, x2 = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    
    x = x1 * (1 - rx) + x2 * rx
    y = y1 * (1 - ry) + y2 * ry
    
    ax.text(x,y,text,fontsize=14)