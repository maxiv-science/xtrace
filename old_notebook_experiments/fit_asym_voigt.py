import numpy as np
import scipy

# model functions for diffraction line fitting
# ack: Gudrun Lotze

def gaussian(x,p):
    y = np.log(2.) * ( (x - p[1])/(0.5*p[2]) )**2
    y = p[0]*np.exp(-y)
    y =  y * 2.*np.sqrt(np.log(2.)/np.pi)/p[2]
    return y

def lorentzian(x,p):
    y = (x - p[1])/(0.5*p[2])
    y = p[0]/(1.+y**2)
    y =  y * 2./np.pi/p[2]
    return y

# (multi-peak) pseudoVoigt function
# parameters = [integI, position, fwhm, LorentzianContent]
# one raw of output matrix per parameters line (one peak in each raw)
# you can sum them by numpy.sum(pseudoVoigt(x,p),0)
def pseudoVoigt(x,p):
    a = np.array(p)
    m = a.size // 4
    a = a.reshape(m,4)
    y = np.zeros((m,len(x)),dtype=np.float32)
    for k in range(0,m):
        if np.absolute(a[k,0])>0.:
            y[k] = (1.-a[k,3])*gaussian(x,a[k,0:3])+a[k,3]*lorentzian(x,a[k,0:3])
    return y

# (multi-peak) pseudoVoigtAsym function
# parameters = [integI, position, fwhm, LorentzianContent, Asymmetry]
# one raw of output matrix per parameters line (one peak in each raw)
# you can sum them by numpy.sum(pseudoVoigtAsym(x,p),0)
def pseudoVoigtAsym(x,p):
    a = np.array(p)
    m = a.size // 5
    a = a.reshape(m,5)
    y = np.zeros((m,len(x)),dtype=np.float32)
    for k in range(0,m):
        if np.absolute(a[k,0])>0.:
            s1 = (1. +    a[k,4])*a[k,2]/2
            s2 = (1. + 1./a[k,4])*a[k,2]/2
            ss = (s1 + s2)/2
            lidx = x < a[k,1]
            y[k, lidx]  = (1.-a[k,3])*s1/ss*gaussian(x[lidx],a[k,0:2].tolist()+[s1])
            y[k, lidx] +=     a[k,3] *s1/ss*lorentzian(x[lidx],a[k,0:2].tolist()+[s1,])
            y[k,~lidx]  = (1.-a[k,3])*s2/ss*gaussian(x[~lidx],a[k,0:2].tolist()+[s2])
            y[k,~lidx] +=     a[k,3] *s2/ss*lorentzian(x[~lidx],a[k,0:2].tolist()+[s2,])
    return y

# model - multiple pseudovoigt with linear background
def model1(x,*p):
    y = np.sum(pseudoVoigt(x,p[0:-2]),0)
    y = y + p[-2]*(x-x[0]) + p[-1]
    return y

# model - multiple pseudoVoigtAsym with linear background
def model1a(x,*p):
    y = np.sum(pseudoVoigtAsym(x,p[0:-2]),0)
    y = y + p[-2]*(x-x[0]) + p[-1]
    return y

def fit(x, y):
    model = model1a
    deg2rad = np.pi/180.
    rad2deg = 180./np.pi
    bkg0 = np.mean( np.concatenate((y[0:3],y[-4:-1])) ) # estimate background
    i0 = np.argmax(y); x0 = x[i0]; y0 = y[i0]-bkg0 # data maximum
    yi = np.trapz(y-bkg0,x)
    fwhm0 = 0.05 #fwhm (fixed)(deg)
    asym0 = 1.0 #asymmetry
    # model parameters
    p0 = [yi, x0, fwhm0*deg2rad, 0.3, asym0, 0.0, bkg0] # [intensity,pos,fwhm,shape,asym,slope,intercept]
    p1, _ = scipy.optimize.curve_fit(model,x,y,p0,sigma=np.sqrt(y),absolute_sigma=True)
    # check shape parameter
    if np.any(p1[3::5]<0.) or np.any(p1[3::5]>1.):
        print('warning: invalid shape parameter')
    fit = model(x,*p1)
    return fit, p1