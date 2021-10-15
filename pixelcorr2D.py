# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:07:35 2021

@author: carlop
"""

from numpy import array, linspace, meshgrid, reshape, concatenate, around, insert
from numpy import zeros, vstack, tile, inner, floor, linalg, where, shape, logical_and
import matplotlib.pyplot as plt


def get_distances(hit_row, hit_col, det_params):    
    def filter(result):
        #Maybe the filtering has to consists on None or Nan
        lidx1 = logical_and(det_xlim[0] <= result[:,0,0], result[:,0,0] <= det_xlim[1])
        lidx2 = logical_and(det_ylim[0] <= result[:,1,0], result[:,1,0] <= det_ylim[1])
        lidx1 = logical_and(lidx1, lidx2)
        lidx2 = logical_and(det_zlim[0] <= result[:,2,0], result[:,2,0] <= det_zlim[1])
        lidx1 = logical_and(lidx1,lidx2)

        return result[lidx1,:,:]
            
    mu = det_params['mu'] 
    I0 = det_params['I0'] 
    
    #pixel dimentions
    pl = det_params['pl']  
    pw = det_params['pw'] 
    ph = det_params['ph'] 
    
    #detector dimentions
    det_numpixels = det_params['det_numpixels'] 
    #detector limits in Cartesian coordinates
    det_xlim = det_params['det_xlim'] 
    det_ylim = det_params['det_ylim'] 
    det_zlim = det_params['det_zlim'] 
    
    #Cartesian coordinates of the sample, given by the poni file 
    sx = det_params['sx'] 
    sy = det_params['sy'] 
    sz = det_params['sz'] 

    j = det_params['j']
    
    numrays = len(hit_col)
    
    hit_x = 0 #for all points
    hit_y = reshape(hit_col * pl, (numrays, 1, 1)) #depth for the different points
    hit_z = reshape(hit_row * ph, (numrays, 1, 1)) 
        
    rayvec_x = zeros((numrays, 1, 1))- sx #for all points
    rayvec_y = reshape(hit_y - sy, (numrays, 1, 1))
    rayvec_z = reshape(hit_z - sz , (numrays, 1, 1))

    #parts of M and y that only depend on the rays and not on the planes
    M_slice = concatenate((concatenate((rayvec_y, -rayvec_x, zeros((numrays, 1, 1))), 2), 
                       concatenate((rayvec_z, zeros((numrays, 1, 1)), -rayvec_x), 2)),
                      1)

    y_slice = concatenate ((((hit_x * rayvec_y) - (hit_y * rayvec_x),
                        (hit_x * rayvec_z) - (hit_z * rayvec_x), )),
                       1)
 
    plt.figure()      
    plt.plot(hit_y[:, 0, 0], hit_z[:, 0, 0], 'o', color = 'blue')
    
    
    result = zeros((numrays, 3, 4*j + 2)) #lets try first with only vh
    i = 0

    #vertical and horizontal intersections
    for planevec in [array([[0,1,0]]), array([[0,0,1]])]:
        for j in range(-j,j+1):
            M = concatenate((M_slice, tile(planevec, (numrays, 1,1))), 1)
            
            if (array([[0,1,0]]) == planevec).all():
                y = concatenate((y_slice, 
                                reshape((floor(hit_col) + j) * pl, (numrays, 1, 1))),
                                1)
                
            elif (array([[0,0,1]]) == planevec).all():
                 y = concatenate((y_slice, 
                                 reshape((floor(hit_row) + j) * ph, (numrays, 1, 1))),   
                                 1)   
             
            result_vh = around(linalg.inv(M) @ (y), 2)
            
            result[:, :, i].reshape((numrays, 3, 1)) = result_vh
            i += 1
    
    
            '''            
            #sorting each ray by x component
            for j in range(numrays):
                result[j, :, : ] = result[j, :, : ].sort(axis = 1)
                print(result[j, :, : ])
            
            
            #Filtering of solutions
            #result_vh = filter(result_vh) #the problem is that now we dont know which filtered solution corresponds to each ray
        
        
            #plt.plot(result_vh[:, 1, 0], result_vh[:, 2, 0], '.', color = 'red')
                    
    #maybe here we have to do the sorting wrt x component and for each ray
    #and after that we could do some filtering    
    
    
    #intersections at the back
    M = concatenate((M_slice, tile(array([[1,0,0]]), (numrays, 1,1))), 1)
    y = concatenate((y_slice, zeros((numrays, 1,1)) + pw), 1)
    result_b = around(linalg.inv(M) @ (y), 2)
    
    
    #Filtering of solutions
    result_b = filter(result_b)
    
    
    #plt.plot(result_b[0, 1, 0], result_b[0, 2, 0], '.', color = 'green')
    plt.plot(result_b[:, 1, 0], result_b[:, 2, 0], '.', color = 'green')

    #next step is to calculate pixels and distances
    #have to sort pixels    
    
    return None
'''
    print(result[0, :, :])
