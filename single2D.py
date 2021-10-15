# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 15:07:35 2021

@author: carlop
"""

from numpy import array, linspace, meshgrid, reshape, concatenate, around
from numpy import zeros, vstack, tile, inner, floor, linalg, where, shape
import matplotlib.pyplot as plt

###############################################################################   
#INFORMATION OF THE SYSTEM 
mu = 3.445930 * 10**-3 #in 1/micron
I0 = 1 

#pixel dimentions
pl = 75 #pixel length, in micrometers 
pw =  450 #pixel width, in micrometers
ph = 75 #pixel height, in micrometers

#detector dimentions
det_numpixels = (100, 100) #number of rows in pixels, number of columns in pixels
#detector limits in Cartesian coordinates
det_xlim = (0.0, pw)
det_ylim = (0.0, det_numpixels[1] * pl)
det_zlim = (0.0, det_numpixels[0] * ph)

#Cartesian coordinates of the sample, given by the poni file 
sx = -0.4051247560395305 * 10 **4 # -dist in poni file, in micrometers
sy = 0.17601077769492505 * 10 **4 # P2 in poni file, in micrometers
sz = 0.02184240399163788 * 10 **4 # P1 in poni file, in micrometers

#for the construction of the plane
j = 10 #positive or negative integer number related to the plane point
#positive means either to the right or upwards
#negative means either to the left or downwards
#planevec
planevec = array([[0,1,0]])

###############################################################################
#information of the frontal intersection point
#hit_col, hit_row and j should be given
hit_col, hit_row = meshgrid(linspace(0, 100, 10), linspace(0, 100, 10)) 
hit_row = hit_row.flatten()
hit_col = hit_col.flatten()
        
hit_x = 0 #for all points
hit_y = reshape(hit_col * pl, (len(hit_col), 1, 1))
hit_z = reshape(hit_row * ph, (len(hit_col), 1, 1)) 
        
rayvec_x = zeros((len(hit_col), 1, 1))- sx #for all points
rayvec_y = reshape(hit_y - sy, (len(hit_col), 1, 1))
rayvec_z = reshape(hit_z - sz , (len(hit_col), 1, 1))

#parts of M and y that only depend on the rays and not on the planes
M_slice = concatenate((concatenate((rayvec_y, -rayvec_x, zeros((len(hit_col), 1, 1))), 2), 
                       concatenate((rayvec_z, zeros((len(hit_col), 1, 1)), -rayvec_x), 2)),
                      1)

y_slice = concatenate ((((hit_x * rayvec_y) - (hit_y * rayvec_x),
                        (hit_x * rayvec_z) - (hit_z * rayvec_x), )),
                       1)
 
plt.figure()  
#plt.axis('square')    
plt.plot(hit_y[:, 0, 0], hit_z[:, 0, 0], 'o', color = 'blue')
#square !!

#vertical and horizontal intersections
for planevec in [array([[0,1,0]]), array([[0,0,1]])]:
    for j in range(-j,j+1):
        M = concatenate((M_slice, tile(planevec, (len(hit_col), 1,1))), 1)
        
        if (array([[0,1,0]]) == planevec).all():
            y = concatenate((y_slice, 
                            reshape((floor(hit_col) + j) * pl, (len(hit_col), 1, 1))),
                            1)
            
        elif (array([[0,0,1]]) == planevec).all():
             y = concatenate((y_slice, 
                             reshape((floor(hit_row) + j) * ph, (len(hit_col), 1, 1))),   
                             1)   
         
        result_vh = around(linalg.inv(M) @ (y), 2)
        
        
        #Still dont know how to filter solutions in a fast way, I need help here
        i = 0
        while(i < len(hit_col)):
            if (not (det_xlim[0] <= result_vh[i, 0, 0] <= det_xlim[1] 
                     and det_ylim[0] <= result_vh[i, 1, 0] <= det_ylim[1] 
                     and det_zlim[0] <= result_vh[i, 2, 0] <= det_zlim[1])):
                result_vh[i, 0, 0] = None
                result_vh[i, 1, 0] = None
                result_vh[i, 2, 0] = None
                #print(i)
            #print(i)
            i += 1
        
        plt.plot(result_vh[:, 1, 0], result_vh[:, 2, 0], '.', color = 'red')

#intersections at the back
M = concatenate((M_slice, tile(array([[1,0,0]]), (len(hit_col), 1,1))), 1)
y = concatenate((y_slice, zeros((len(hit_col), 1,1)) + pw), 1)
result_b = around(linalg.inv(M) @ (y), 2)

#Still dont know how to filter solutions in a fast way, I need help here
i = 0
while(i < len(hit_col)):
    if (not (det_xlim[0] <= result_b[i, 0, 0] <= det_xlim[1] 
             and det_ylim[0] <= result_b[i, 1, 0] <= det_ylim[1] 
             and det_zlim[0] <= result_b[i, 2, 0] <= det_zlim[1])):
        result_b[i, 0, 0] = None
        result_b[i, 1, 0] = None
        result_b[i, 2, 0] = None
        #print(i)
    #print(i)
    i += 1

plt.plot(result_b[:, 1, 0], result_b[:, 2, 0], '.', color = 'green')