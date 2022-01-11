#from numpy import array, linspace, meshgrid, reshape, concatenate, around
#from numpy import zeros, vstack, tile, inner, floor, linalg, where, shape
#import numpy as np
from cupy import array, linspace, meshgrid, reshape, concatenate, around
from cupy import zeros, vstack, tile, inner, floor, linalg, where, shape
import cupy as np
import cupy
import matplotlib.pyplot as plt
import cupyx.scipy.sparse as cusp
import cupyx
import time

# solve system of 3x3 equations using Cramer's rule

def cramers_rule(a,b):
    def determinant(a):
        return a[:,0,0]*(a[:,1,1]*a[:,2,2]-a[:,1,2]*a[:,2,1])-a[:,0,1]*(a[:,1,0]*a[:,2,2]-a[:,1,2]*a[:,2,0])+a[:,0,2]*(a[:,1,0]*a[:,2,1]-a[:,1,1]*a[:,2,0])
    n = a.shape[0]
    # preallocate space for the result
    x = np.nan * np.ones((n,3),dtype=np.float32)
    det = determinant(a)
    idx = det.nonzero()[0]
    aa = a.copy()
    aa[:,:,0] = b
    x[idx,0] = determinant(aa)[idx]/det[idx]
    aa = a.copy()
    aa[:,:,1] = b
    x[idx,1] = determinant(aa)[idx]/det[idx]
    aa = a.copy()
    aa[:,:,2] = b
    x[idx,2] = determinant(aa)[idx]/det[idx]
    return x, idx


def sensor_depth_spill_psf(params, hit_row, hit_col):
    #do_plot = False # set false for 1M rays
    startTime = time.time()
    
    hit_x = 0 #for all points
    hit_y = reshape(hit_col * params["pl"], (hit_col.size, 1, 1))
    hit_z = reshape(hit_row * params["ph"], (hit_col.size, 1, 1)) 
    
    hit_ids = np.arange(hit_col.size)
    
    rayvec_x = zeros((len(hit_col), 1, 1)) - params["sx"] #for all points
    rayvec_y = reshape(hit_y - params["sy"], (len(hit_col), 1, 1))
    rayvec_z = reshape(hit_z - params["sz"] , (len(hit_col), 1, 1))

    #parts of M and y that only depend on the rays and not on the planes
    M_slice = concatenate((concatenate((rayvec_y, -rayvec_x, zeros((len(hit_col), 1, 1))), 2), 
                        concatenate((rayvec_z, zeros((len(hit_col), 1, 1)), -rayvec_x), 2)),
                        1)
    
    y_slice = concatenate ((((hit_x * rayvec_y) - (hit_y * rayvec_x),
                            (hit_x * rayvec_z) - (hit_z * rayvec_x), )),
                        1)
    
    #if do_plot:
    #    plt.figure(figsize=(10,8))       
    #    plt.plot(cupy.asnumpy(hit_y[:, 0, 0])/params["pl"], cupy.asnumpy(hit_z[:, 0, 0])/params["ph"], 'o', color = 'blue')
    #    plt.axis('equal')

    # buffer for intersections
    Jjmax = 2*params["Jj"]
    result_buf = np.nan * np.ones((hit_col.size,Jjmax,3),dtype=np.float32)
    result_cnt = np.ones((hit_col.size,),dtype=np.int32)
    # store the frontal plane in the result buffer
    result_buf[:,0,0] = hit_x
    result_buf[:,0,1] = hit_y[:,0,0]
    result_buf[:,0,2] = hit_z[:,0,0]

    #vertical and horizontal intersections
    for planevec in [array([[0,1,0]]), array([[0,0,1]])]:
        for j in range(-params["Jj"],params["Jj"]+1):
            M = concatenate((M_slice, tile(planevec, (len(hit_col), 1,1))), 1)
            
            if (array([[0,1,0]]) == planevec).all():
                y = concatenate((y_slice, 
                                reshape((floor(hit_col) + j) * params["pl"], (len(hit_col), 1, 1))),
                                1)
                
            elif (array([[0,0,1]]) == planevec).all():
                y = concatenate((y_slice, 
                                reshape((floor(hit_row) + j) * params["ph"], (len(hit_col), 1, 1))),   
                                1)   
            
            # solving with Cramer's rule
            nhits = hit_col.size
            # preallocate space for the result
            result_vh = np.nan * np.ones((nhits,3),dtype=np.float32)
            #result_vh = around(linalg.inv(M) @ (y), 2)
            y = np.squeeze(y, axis=2) # note: need to squeeze the last dimension
            result_vh, idx = cramers_rule(M,y)
            result_vh = np.expand_dims(result_vh, axis=2) # note: unsqueeze
            #result_vh = around( result_vh, 2) # not sure what it is doing
            
            # results may contain nan-solutions, valid soutions are at idx, it can be used as a filter
            result_vh = result_vh[idx,:,:]
            ids = hit_ids[idx]
            
            #lidx1 = np.logical_and(det_xlim[0]<=result_vh[:,0,0], result_vh[:,0,0]<=det_xlim[1])
            #lidx2 = np.logical_and(det_ylim[0]<=result_vh[:,1,0], result_vh[:,1,0]<=det_ylim[1])
            #lidx1 = np.logical_and(lidx1,lidx2)
            #lidx2 = np.logical_and(det_zlim[0]<=result_vh[:,2,0], result_vh[:,2,0]<=det_zlim[1])
            #lidx1 = np.logical_and(lidx1,lidx2)
            
            #result_vh = result_vh[lidx1,:,:]
            
            lidx  = (params["det_xlim"][0]<=result_vh[:,0,0]) * (result_vh[:,0,0]<=params["det_xlim"][1]) * \
                    (params["det_ylim"][0]<=result_vh[:,1,0]) * (result_vh[:,1,0]<=params["det_ylim"][1]) * \
                    (params["det_zlim"][0]<=result_vh[:,2,0]) * (result_vh[:,2,0]<=params["det_zlim"][1])
            
            result_vh = result_vh[lidx,:,:]
            ids = ids[lidx]
            
            #if do_plot:
            #    plt.plot(cupy.asnumpy(result_vh[:, 1, 0])/pl, cupy.asnumpy(result_vh[:, 2, 0])/params["ph"], '.', color = 'red')
            
            # store the result
            result_cnt[ids] += 1
            lidx = result_cnt[ids]<=Jjmax
            if not np.all(lidx):
                print('warning: have more intersections than assumed')
                result_cnt[ids[~lidx]] = Jjmax
                #_t = result_vh[result_cnt[ids]>Jjmax,1:,0]
                #_t[:,0] /= pl
                #_t[:,1] /= ph
                #print(_t)
            result_buf[ids[lidx],result_cnt[ids[lidx]]-1,:] = result_vh[lidx,:,0]

    #intersections at the back
    M = concatenate((M_slice, tile(array([[1,0,0]]), (len(hit_col), 1,1))), 1)
    y = concatenate((y_slice, zeros((len(hit_col), 1,1)) + params["pw"]), 1)
    #result_b = around(linalg.inv(M) @ (y), 2)
    y = np.squeeze(y, axis=2) # note: need to squeeze the last dimension
    result_b, idx = cramers_rule(M,y)       
    result_b = np.expand_dims(result_b, axis=2) # note: unsqueeze
    #result_b = around( result_b, 2) # not sure what it is doing

    result_b = result_b[idx,:,:]
    ids = hit_ids[idx]

    lidx  = (params["det_xlim"][0]<=result_b[:,0,0]) * (result_b[:,0,0]<=params["det_xlim"][1]) * \
            (params["det_ylim"][0]<=result_b[:,1,0]) * (result_b[:,1,0]<=params["det_ylim"][1]) * \
            (params["det_zlim"][0]<=result_b[:,2,0]) * (result_b[:,2,0]<=params["det_zlim"][1])

    # Note we can get e.g. the corner [0,0,0] exit point twice, i.e. from different type of planes

    result_b = result_b[lidx,:,:]
    ids = ids[lidx]

    # store the result
    result_cnt[ids] += 1
    lidx = result_cnt[ids]<=Jjmax
    if not np.all(lidx):
        print('warning: have more intersections than assumed')
        result_cnt[ids[~lidx]] = Jjmax
        #_t = result_vh[result_cnt[ids]>Jjmax,1:,0]
        #_t[:,0] /= pl
        #_t[:,1] /= ph
        #print(_t)
    result_buf[ids[lidx],result_cnt[ids[lidx]]-1,:] = result_b[lidx,:,0]

    # sort results by x-coordinate
    ind = np.argsort(result_buf[:,:,0],axis=1)
    result_buf = np.take_along_axis(result_buf, ind[:,:,np.newaxis], axis=1)

    # calculate distances
    dist = np.sqrt(np.sum(np.square(result_buf[ids,1:,:] - result_buf[ids,:-1,:]), axis=2))
    # calculate centres
    cents = (result_buf[ids,1:,:] + result_buf[ids,:-1,:])/2

    # calculate absorption and transmission coefficients
    ti = np.exp(-params["mu"]*dist)
    ai = 1-ti
    ti = np.cumprod(ti,axis=1)
    ai[:,1:] *= ti[:,:-1]

    ai *= params["IO"]
    ti *= params["IO"]

    # collect results
    lidx = np.arange(Jjmax-1)[np.newaxis,:]*np.ones((ai.shape[0],1)) < (result_cnt[ids,np.newaxis]-1)
    col_idx = np.floor(hit_row[ids]).astype(int) *  params["det_numpixels"][1] + np.floor(hit_col[ids]).astype(int)
    col_idx = (np.ones(ai.shape,dtype=np.int64)*col_idx[:,np.newaxis])[lidx]
    ai = ai[lidx]
    ti = ti[lidx] 
    row_idx = (cents/np.array([params["pw"],params["pl"],params["ph"]])).astype(int) # index of the center
    row_idx = row_idx[:,:,2] * params["det_numpixels"][1] + row_idx[:,:,1] # 2d-index to flatten index
    row_idx = row_idx[lidx]

    # get resut to cpu (seems not work always)
    #print(dist[2,0])
    #print(result_b[0,:,:])

    sl=slice(0,6)
    #print(col_idx[sl])
    #print(row_idx[sl])
    #print(ai[sl])

    executionTime = (time.time() - startTime)
    #print('nhits:', nhits, ', exec. time (sec):', executionTime)
    
    dim = params["det_numpixels"]
    N = dim[0]*dim[1]
    G = cusp.csr_matrix((ai, (row_idx, col_idx)), shape=(N,N))
    return G

def regularized_richard_lucy_deconv(img, psf, smooth_coeff, iters):
    pixel_arr = np.array(img).reshape(-1)
    pixel_arr_itr = pixel_arr.copy()
    for _ in range(iters):
        c = psf@pixel_arr_itr
        c[c == 0] = 1
        grad = np.gradient(pixel_arr_itr.reshape(img.shape))
        divergence = np.gradient(grad[0], axis=0) + np.gradient(grad[1], axis=1)
        divergence_arr = divergence.reshape(-1)
        divergence_arr /= np.sum(np.abs(divergence_arr))
        sm = psf.T@(pixel_arr/c)
        temp_1 = pixel_arr_itr*sm
        temp_2 = (1 - smooth_coeff*divergence_arr)
        pixel_arr_itr = temp_1/temp_2
    return pixel_arr_itr.reshape(img.shape)