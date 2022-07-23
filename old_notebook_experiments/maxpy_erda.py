from __future__ import print_function # temporarily
import numpy as np
import re, sys, time, os

PLOG_USELAST   = 0
PLOG_INTERP    = 1

#---------------------------------------------------------------------
# this thing is here only temporarily

class ProgressBar:
    def __init__(self, iterations):
        self.iterations = iterations
        self.prog_bar = '[]'
        self.fill_char = '*'
        self.width = 50
        self.__update_amount(0)

    def animate(self, iter):
        print('\r', self, end='')
        sys.stdout.flush()
        self.update_iteration(iter)
        print('\r', self, end='')
        sys.stdout.flush()

    def update_iteration(self, elapsed_iter):
        self.__update_amount((elapsed_iter / float(self.iterations)) * 100.0)
        self.prog_bar += '  %d of %s complete' % (elapsed_iter, self.iterations)

    def __update_amount(self, new_amount):
        percent_done = int(round((new_amount / 100.0) * 100.0))
        all_full = self.width - 2
        num_hashes = int(round((percent_done / 100.0) * all_full))
        self.prog_bar = '[' + self.fill_char * num_hashes + ' ' * (all_full - num_hashes) + ']'
        pct_place = (len(self.prog_bar) // 2) - len(str(percent_done))
        pct_string = '%d%%' % percent_done
        self.prog_bar = self.prog_bar[0:pct_place] + \
            (pct_string + self.prog_bar[pct_place + len(pct_string):])

    def __str__(self):
        return str(self.prog_bar)
#---------------------------------------------------------------------

# tth = tth2Dsimple(delta,N,M,params) - calculate diffraction 2Theta (rad)
#     for (N,M) matrices of pixels for a 2D detector rotated by angle
#     delta. Detector parameters are stored in the associative array:
#     
#     params = { 'n0': 0, 'm0': 0, 'wn': width(n)/L, 'wm': width(m)/L, 'phi': 0. }
def tth2Dsimple(delta,N,M,params):
    # get parameters
    n0 = params['n0'] # n0 - detector zero
    m0 = params['m0'] # m0 - detector zero
    wn = params['wn'] # wn/L
    wm = params['wm'] # wm/L
    phi = params['phi'] # rotation around detector axis (optimize)
    # calculate pixel coordinates in the lab ref. system
    # apply detector phi-rotation
    c = np.cos(phi)
    s = np.sin(phi)
    tN = c*(N-n0)*wn - s*(M-m0)*wm
    tM = s*(N-n0)*wn + c*(M-m0)*wm
    # main axis rotation
    c = np.cos(delta)
    s = np.sin(delta)
    X = c - s*tN
    Y = s + c*tN
    Z = tM
    R = np.sqrt( X**2 + Y**2 + Z**2 )
    tth = np.arccos(X/R)
    return tth

# tth = tth2Dwithtilt(delta,N,M,params) - calculate diffraction 2Theta (rad)
#     for (N,M) matrices of pixels for a 2D detector rotated by angle
#     delta. Detector parameters are stored in the associative array:
#     
#     params = { 'n0': 0, 'm0': 0, 'wn': width(n)/L, 'wm': width(m)/L, 'phi': 0., 'rot': 0., 'tilt': 0. }
def tth2DwithTilt(delta,N,M,params):
    # get parameters
    n0 = params['n0'] # n0 - detector zero
    m0 = params['m0'] # m0 - detector zero
    wn = params['wn'] # wn/L
    wm = params['wm'] # wm/L
    phi = params['phi'] # rotation around detector axis (optimize)
    rot  = params['rot'] # Fit2D tilt-plane rotation parameter
    tilt = params['tilt'] # Fit2D detector tilt
    # calculate pixel coordinates in the lab ref. system
    # apply detector phi-rotation
    c = np.cos(phi)
    s = np.sin(phi)
    tN = c*(N-n0)*wn - s*(M-m0)*wm
    tM = s*(N-n0)*wn + c*(M-m0)*wm
    # apply detector tilt
    W = tiltMatrixFit2D(rot,tilt)
    tX = W[0,1]*tN + W[0,2]*tM + 1. # including distance of beam center on the detector from the sample
    tY = W[1,1]*tN + W[1,2]*tM
    tZ = W[2,1]*tN + W[2,2]*tM
    # note: up to now the rotation matrix is constant
    # i.e. can be saved and does not need to be recalculated every time (optimize)
    # main axis rotation
    c = np.cos(delta)
    s = np.sin(delta)
    X = c*tX - s*tY
    Y = s*tX + c*tY
    Z = tZ
    # calculate 2Theta
    R = np.sqrt( X**2 + Y**2 + Z**2 )
    tth = np.arccos(X/R)
    return tth

# R = rotMatrix(theta, axis) - matrix representing clockwise rotation around
#                              the given axis by angle theta (in radians)
def rotMatrix(theta=0.0, axis=[0,0,1]):
    # normailsed axis coordinates
    w = np.array(axis)
    w = w/np.linalg.norm(w)
    # Rodrigues' rotation matrix
    Om = np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    R = np.eye(3) + Om * np.sin(theta) + np.dot(Om,Om) * (1.-np.cos(theta))
    return R

# R = rotMatrixFit2D(theta, axis) - matrix representing detector tilt with
#                                   respect to direct beam with Fit2D notation
def tiltMatrixFit2D(rot,tilt):
    axis = np.array([0,-np.cos(rot),np.sin(rot)])
    R = rotMatrix(-tilt,axis)
    return R

# maxval, i1, i2 = findmax2(A) - returns maximum value and its indices
def findmax2(A):
    n, m = A.shape # reshape
    B = A.reshape(n*m)
    mval = max(B) # find maximum value
    idxs = [idx for idx, val in enumerate(B) if val == mval] # find indexes
    idxs1 = [idx / m for idx in idxs] # convert to 2d indexes
    idxs2 = [idx % m for idx in idxs]
    return mval, idxs1, idxs2

# a,b,ea,eb = linfit(x,y,ey) - linear regression of data with y-errorbars (a-slope) 
def linfit(x,y,err):
    Sx = 0.; Sy = 0.; S = 0.
    for idx, ee in enumerate(err):
        ee2 = ee*ee
        Sx += x[idx]/ee2
        Sy += y[idx]/ee2
        S += 1./ee2
    Stt = 0.; slope = 0.
    for idx in range(0,len(x)):
        t = (x[idx]-Sx/S)/err[idx]
        slope += t*y[idx]/err[idx]
        Stt += t*t
    slope *= 1./Stt
    intercept = (Sy-Sx*slope)/S
    eslope = np.sqrt(1./Stt)
    eintercept = np.sqrt(1./S*(1.+Sx*Sx/S/Stt))
    return slope, intercept, eslope, eintercept

# pos,delay,mot,n = read_plog(filename) - read motor positions log file
def read_plog(filename):
    delay = []; pos = []; imgn = []; mot = 'none'; n = 0
    
    r = re.compile('[/ \t\n\r:]+')
    
    f = open(filename, 'r')
    ii = 0
    rec = []
    for line in f:
        #print line,
        line = str.lstrip(line)
        if (len(line)>0 and line[0]=='#'):
            continue
        rec = r.split(line) # i/n  delay(sec)  motor  pos
        imgn.append( int(rec[0]) )
        delay.append( float(rec[2]) )
        pos.append( float(rec[4]) )
        ii += 1

    f.close()
    if (ii>0):
        n = int( rec[1] )
        mot = rec[3]
    return pos, delay, imgn, mot, n

# cpos,cdelay =  complement_plog_data(pos, delay, imgn, n, metod) - correct/complement missing records
def complement_plog_data(pos, delay, imgn, n, corr=PLOG_INTERP):
    cpos = [float(0.0)] * (imgn[-1]+1); cdelay = [float(0.0)] * (imgn[-1]+1)
    bexIdx = []
    ii = 0
    for irec in range(0,len(pos)):
        if (ii==imgn[irec]):
            cpos[ii]=pos[irec]
            cdelay[ii]=delay[irec]
            ii = ii + 1
        elif (ii<imgn[irec]): # record(s) missing
            nmiss = imgn[irec]-ii
            print("(warning) maxpy.complement_plog_data: missing records(s) (%d:%d)" % (ii,imgn[irec]-1,))
            if (corr==PLOG_USELAST):
                cpos[ii:imgn[irec]] = [pos[irec]]*(nmiss+1)
                cdelay[ii:imgn[irec]] = [delay[irec]]*(nmiss+1)
            elif (corr==PLOG_INTERP):
                p0 = cpos[ii-1] if (ii>0) else pos[irec]
                p1 = pos[irec]
                d0 = cdelay[ii-1] if (ii>0) else delay[irec]
                d1 = delay[irec]
                for jj in range(ii,imgn[irec]+1):
                    cpos[jj] = p0 + (p1-p0)*(jj-ii)/nmiss
                    cdelay[jj] = d0 + (d1-d0)*(jj-ii)/nmiss
            else:
                raise ValueError('Unknown method', corr)
            ii = ii + nmiss+1
        elif (ii==imgn[irec]+1): # excessive record
            print("(warning) maxpy.complement_plog_data: excessive record (irec:%d, imgn[irec]:%d)" % (irec,imgn[irec],))
            # we need to repair a whole segment - skip/remove this record and mark for correction
            bexIdx.append(ii)
        else:
            print("(error) maxpy.complement_plog_data: unknow error in plog (irec:%d, imgn[irec]:%d)" % (irec,imgn[irec],))
            raise NotImplementedError("Unknown error in plog.")
    # correct segments containing excessive records
    if len(bexIdx)>0:
        print("(warning) maxpy.complement_plog_data: Interpolating broken data in vicinity of exccessive records(s).")
    for ib in bexIdx:
        nhalf = 10
        idx = []
        if (ib>=nhalf and ib<=len(cpos)-nhalf): # somewhere in the middle
            idx = range(ib-nhalf,ib+nhalf)
        elif (ib+2*nhalf<=len(cpos)): # close to beginning but far from end
            idx = range(0,ib+2*nhalf)
        elif (ib>=2*nhalf): # close to end but far from beginning
            idx = range(ib-2*nhalf,len(cpos))
        else: # short data
            idx = range(0,len(cpos))
        for irep in range(0,2): # two steps method
            # mean relative difference between delays (using linfit)
            a,b,ea,eb = linfit(idx,[cdelay[i] for i in idx],[1.0]*len(idx))
            # find broken and correct points
            bidx = []; cidx = []
            if irep==0:
                err = 0.5*abs(a) if 0.5*abs(a)>0.005 else 0.005
            else:
                err = 0.1*abs(a) if 0.1*abs(a)>0.005 else 0.005
            for i in idx:
                if abs(cdelay[i]-(a*i+b))<err:
                    cidx.append(i)
                else:
                    bidx.append(i)
            # linear coefficients from good points
            adel,bdel,eadel,ebdel = linfit(cidx,[cdelay[i] for i in cidx],[1.0]*len(cidx))
            apos,bpos,eapos,ebpos = linfit(cidx,[cpos[i] for i in cidx],[1.0]*len(cidx))
            # interpolation in broken points
            for i in bidx:
                cdelay[i]=adel*i+bdel
                cpos[i]=apos*i+bpos
    return cpos, cdelay

# y, n, img = get_imgline(fileName,roi2,imask) - reads image and returns dataline summed
#     over roi2 as y, number of pixels contributing to each y-point as n and image
#     corrected for bad pixels as img
#
# usage: imask = ((136, 82, 56),(203,190,101)) # bad pixels
#        roi2 = range(m0-10,m0+11) # summation roi
#        y,n,img = get_imgline(fileName,roi2,imask)
def get_imgline(fileName,roi2,imask):
    # load image
    edfData = PyMca.PyMcaIO.EdfFile.EdfFile(FileName=fileName,access='rb', fastedf=True)
        
    img = edfData.GetData(0)
    m, n = img.shape
        
    # zero wrong pixels
    img[imask[0],imask[1]] = 0
    mask = np.ones((m,n),int)
    mask[imask[0],imask[1]] = 0
    
    # integrate
    y = sum(img[roi2,:],0)
    c = sum(mask[roi2,:],0)
    
    return y,c,img

def ttheq_get_indexes(xtth,delta,detParams,N=[],M=[],tthFnc=tth2Dsimple):
    # generate indexes into image
    bng = False # new grid
    if len(N)==0:
       N = np.arange(0,detParams['n'],1,int)
       bng = True
    if len(M)==0:
       M = np.arange(0,detParams['m'],1,int)
       bng = True
    if bng:
        M,N = np.meshgrid(M,N) # image notation
    # get tth
    TTH = tthFnc(delta,N,M,detParams)
    # convert tth values to indexes into xtth
    xstep = xtth[1]-xtth[0]
    xmin = xtth[0] - xstep/2
    idx = (TTH - xmin)/xstep + 0.5 # calculation
    lidx = (idx>=0) * (idx<len(xtth)) # this must be before rounding
    idx = idx.astype(int) # rounding
    return idx, lidx

def ttheq_get_indexes_withPixelSplitting(xtth,delta,detParams,N=[],M=[],tthFnc=tth2Dsimple):
    # generate indexes into image
    bng = False # new grid
    if len(N)==0:
       N = np.arange(0,detParams['n'],1,int)
       bng = True
    if len(M)==0:
       M = np.arange(0,detParams['m'],1,int)
       bng = True
    if bng:
        M,N = np.meshgrid(M,N) # image notation
    # get tth
    TTH = tthFnc(delta,N,M,detParams)
    # convert tth values to indexes into xtth
    xstep = xtth[1]-xtth[0]
    xmin = xtth[0] - xstep/2
    idx = (TTH - xmin)/xstep + 0.5 # calculation
    # get tth-range
    dTTHN = tthFnc(delta,N+1,M,detParams)-TTH
    dTTHM = tthFnc(delta,N,M+1,detParams)-TTH
    widx = np.sqrt(dTTHN*dTTHN + dTTHM*dTTHM)/xstep # index width
    # quite ineffective method (find the most broad pixel-2Theta-half-width)
    hfwidx = int( np.ceil( np.amax(widx)/2. ) )
    exIdx = np.empty(shape=(idx.shape+(1+2*hfwidx,)),dtype=float) # extended indexes
    exW = np.zeros(shape=(idx.shape+(1+2*hfwidx,)),dtype=float) # weights
    for k in range(-hfwidx,0):
       exIdx[:,:,k+hfwidx] = idx + k
       lIdx1 = -k+0.5 <= widx/2.
       exW[lIdx1,k+hfwidx] = 1./widx[lIdx1]
       lIdx2 = np.logical_and( -k+0.5 > widx/2., -k-0.5 < widx/2. )
       exW[lIdx2,k+hfwidx] = 0.5-(-k-0.5)/widx[lIdx2]
    exIdx[:,:,hfwidx] = idx
    exW[:,:,hfwidx] = 1./widx
    for k in range(1,hfwidx+1):
       exIdx[:,:,k+hfwidx] = idx + k
       lIdx1 = k+0.5 <= widx/2.
       exW[lIdx1,k+hfwidx] = 1./widx[lIdx1]
       lIdx2 = np.logical_and( k+0.5 > widx/2., k-0.5 < widx/2. )
       exW[lIdx2,k+hfwidx] = 0.5-(k-0.5)/widx[lIdx2]
    lidx = np.logical_or(exIdx<0, exIdx>=len(xtth)) # this must be before rounding
    exW[lidx] = 0
    # remove culumns containing only zero weights
    clidx = np.any(exW>0,axis=(0,1))
    exW = exW[:,:,clidx]
    exIdx = exIdx[:,:,clidx]
    lidx = lidx[:,:,clidx]
    exIdx = exIdx.astype(int) # rounding
    return exIdx, exW

def optimal_bin_width(delta,detParams,N=[],M=[],tthFnc=tth2Dsimple):
    # generate indexes into image
    bng = False # new grid
    if len(N)==0:
       N = np.arange(0,detParams['n'],1,int)
       bng = True
    if len(M)==0:
       M = np.arange(0,detParams['m'],1,int)
       bng = True
    if bng:
        M,N = np.meshgrid(M,N) # image notation
    # get tth
    TTH = tthFnc(delta,N,M,detParams)
    # get tth-range
    dTTHN = tthFnc(delta,N+1,M,detParams)-TTH
    dTTHM = tthFnc(delta,N,M+1,detParams)-TTH
    widx = np.sqrt(dTTHN*dTTHN + dTTHM*dTTHM) # index width (with xstep=1)
    return np.percentile(widx,80)
    
def export_specfile(filename, x, y, xlabel='x', ylabel=None, legend=None):
    # check input parameters
    if not ylabel:
        if len(y.shape)>1:
            ylabel = []
            for i in range(y.shape[0]):
                ylabel.append( 'det{:d}'.format(i) )
        else:
            ylabel = 'det'
    # open file
    ffile=open(filename,'wb')
    ffile.write("#F %s\n" % filename)
    savingDate = "#D %s\n"%(time.ctime(time.time()))
    ffile.write(savingDate)
    ffile.write("\n")
    ffile.write("#S 1 %s\n" % legend)
    ffile.write(savingDate)
    ffile.write("#N %d\n" % (len(y.shape)+1))
    if len(y.shape)>1:
        ffile.write("#L %s" % (xlabel))
        for i in range(y.shape[0]):
            ffile.write("  %s" % (ylabel[i]) )
        ffile.write("\n")
        for i in range(y.shape[1]):
            ffile.write("%.7g" % (x[i]))
            for yval in y[:,i]:
                ffile.write("  %.7g" % (yval) )
            ffile.write("\n")
    else:
        ffile.write("#L %s  %s\n" % (xlabel, ylabel) )
        for i in range(len(y)):
            ffile.write("%.7g  %.7g\n" % (x[i], y[i]))
    ffile.write("\n")
    #close file
    ffile.close()

# idx, xcorr = checkDataSpacingLinear(x,reldev,absdev) - check linear spacing in x
#     Check linear spacing in vector x and returns indexes of outliers and a copy
#     of x interpolated in these points.
#     Relative or absolute deviation can be specified. Use None to ignore
#     the option.
def checkDataSpacingLinear(x,reldev=3.,absdev=None):
    # linear regression
    n = np.arange(x.size,dtype=np.float)
    # a little 'robust' initial estimation
    a = np.median(x[2:]-x[1:-1])
    b = np.median(x-a*n)
    # removing outliers
    dif = x-(a*n+b)
    idx = np.where( np.absolute(dif)<=3.*np.std(dif) )
    nn = n[idx]; xx = x[idx] # reliable points
    # fitting
    a, b = np.polyfit(nn,xx,1)
    # absolute difference
    dif = np.absolute(a*n+b - x)
    # bad data indexes
    lidx = np.zeros(x.shape,dtype=np.bool)
    if reldev:
        lidx = np.logical_or(dif > reldev*a, lidx)
    if absdev:
        lidx = np.logical_or(dif > absdev, lidx)
    idx = np.where(lidx)[0]
    # corrected values
    xcorr = np.copy(x)
    xcorr[idx] = a*idx + b
    return idx, xcorr

# strformat, i0 = numberedFileNameFormat(filename): get string formatter for an image file
#       Return string formatter and img number from a given fileneme.
#       Example: filename = "/mydata/dir_1/dir_2/scanfile_002.edf" gives
#         -> strformat = /mydata/dir_1/dir_2/scanfile_{:03d}.edf
#         -> i0 = 2
def numberedFileNameFormat(filename):
    filebase, file_ext = os.path.splitext(filename)
    filebase, filenb = filebase.rsplit("_",1)
    i0, dlen = int(filenb), len(filenb)
    strformat = filebase + ("_{:0%dd}" % (dlen,)) + file_ext
    return strformat, i0

# --- Hdf5 buffered image writer ---------------------------------------------------------------
class h5BufferedImgFileWriter:
    fid = [] # hdf5 file identifier

    bufflen = 0 # img buffer length
    
    imgBuffer = [] # [bufferlen x n x m] image buffer
    imgNbBuffer = [] # bufferlen array of image numbers
    
    def __init__(self,fid,nimages,imgshape,name='images',bufferlen=100):
        self.fid = fid
        self.bufflen = bufferlen
        n = imgshape[0]
        m = imgshape[1]
        # prepare buffer
        self.imgBuffer = np.empty([bufferlen,n,m],dtype=np.int32)
        self.imgNbBuffer = np.zeros([bufferlen,1],dtype=np.int32)
        self.nbuf = 0
        # create dataset
        self.dset = self.fid.create_dataset("data/images", (nimages,n,m), maxshape=(None,n,m),dtype='i4',compression="lzf", chunks=(1,n,m))
    
    def add_image(self,imgnb,img):
        if (self.bufflen<=1):
            self.dset[imgnb,:,:] = img # no buffering
            return
        # safety check    
        if (self.nbuf>=self.bufflen):
            raise Exception("Full data buffer.")
        # add to buffer    
        self.imgNbBuffer[self.nbuf] = imgnb
        self.imgBuffer[self.nbuf] = img
        self.nbuf = self.nbuf+1
        # flush
        if (self.nbuf>=self.bufflen):
            self.flush()
    
    def flush(self):
        # save data into hdf5
        if (self.nbuf>0):
            self.dset[self.imgNbBuffer[0:self.nbuf],:,:] = self.imgBuffer[0:self.nbuf,:,:]
            self.nbuf = 0
    
    def __del__(self):
        if (self.nbuf>0):
            print('Warning: (h5BufferedImgFileWriter:Destructor) Buffer not empty.')
