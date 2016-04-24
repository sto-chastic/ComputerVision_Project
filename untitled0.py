# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:52:04 2016

@author: David
"""

import csv 
import cv2
import numpy as np
from numpy import linalg as LA
import fnmatch
import os
import matplotlib.pyplot as plt



def rescale(A):
    [c,r] = A.shape
    mean = A[0,:];#Assign the first image as the MEAN image
    error = 10;
    test = np.zeros((1,80));
    old = np.zeros((1,80))
    j=0
    while error>0.0000001:
        scale = np.sqrt(np.sum(np.square(mean)))
        
        A = np.divide(A,scale)
        mean = np.divide(mean,scale)
        old[:] = mean[:];
        current = np.zeros((2,40))
        
        for i in range(c):#For for calculating the rotation of the matrices.
            current[0,:] = A[i,::2];#Divide in X coordinates
            current[1,:] = A[i,1::2];#Divide in Y coordinates
            alpha = (np.dot(current[0,:],mean[1::2])-np.dot(current[1,:],mean[0::2]))/np.dot(mean,A[i,:])
            alpha = np.arctan(alpha)#Optimal angle of rotation is found.
            
            tangent = np.dot(mean,A[i,:])
            plt.plot(A[i,::2],A[i,1::2])
            plt.plot(mean[::2],mean[1::2])
            T = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])#Calculating the rotation matrix
            A[i,::2] = np.dot(T,current)[0,:]/tangent;#Rotating the matrix X coordinate
            A[i,1::2] = np.dot(T,current)[1,:]/tangent;#Rotating the matrix Y coordinate
            plt.plot(A[i,::2],A[i,1::2])
            plt.show()
            
        mean[:] = np.mean(A,0)#Create a new mean matrix based on the mean of the rotated and scaled shapes.
        scale = np.sqrt(np.sum(np.square(mean)));
        mean[:] = mean /scale;
        error = np.linalg.norm(mean-old);
        print j
        print error
    return mean,A,error

def PCA(X,Variation):
    '''
    Do a PCA analysis on X
    @param X:                np.array containing the samples
                             shape = (nb samples, nb dimensions of each sample)
    @param Variance:         Proportion of the total variation desired.                        
    @return: return the nb_components largest eigenvalues and eigenvectors of the covariance matrix and return the average sample 
    '''
    [n,d] = X.shape
    
    
    Xm = np.mean(X, axis=0)
    x = np.zeros((n,d))
    for i in range(0,n):
        for j in range(0,d):
            x[i,j] = X[i,j]-Xm[j]
    x = x.T
    
    Xc = np.dot(x.T,x)
    
    L,V = LA.eig(Xc)
    
    [ne] = L.shape
    
    index = np.argsort(-L)
    
    Li = L[index]
    
    varTot = np.sum(Li)
    varSum = 0;
    for numEig in range(0,ne):
        varSum = varSum + Li[numEig]
        print varSum
        print varTot
        if varSum/varTot >= Variation:
            print 'Number of Eigenvectors after PCA ='
            print numEig
            break
    
    Vi = np.dot(x,V)
    
    
    Vii = Vi[:,index]
    Viii = Vii[:,:numEig]
    Liii = Li[:numEig]
    
    
    for ii in range(0,numEig):
        Viii[:,ii] = Viii[:,ii]/sum(Viii[:,ii])

    print Viii.shape
    return [Liii,Viii,Xm]
    
    
def Matching(target,eigVals,eigVecs,mean):
    '''
    @param target:                Target shape
    '''
    
    stop = 0
    b = np.zeros((1,80))
    current = np.zeros((2,40))
    error = 10
    
    Xt = np.mean(target[1,::2]);#Divide in X coordinates
    Yt = np.mean(target[1,1::2]);#Divide in Y coordinates
    target[i,::2]  = target[i,::2]-np.mean(target[i,::2]);#Zero-mean of the X axis
    target[i,1::2] = target[i,1::2]-np.mean(target[i,1::2]);#Zero-mean of the Y axis
    
    while error > 0.0001:#change later, the error is not thresholded, it stops when the error doesn't change significantly in various iterations
        xIni = mean + np.dot(eigVecs,b)
        
        current[0,:] = xIni[1,::2];#Divide in X coordinates
        current[1,:] = xIni[1,1::2];#Divide in Y coordinates
        b = (np.dot(current[0,:],target[1::2])-np.dot(current[1,:],target[0::2]))/np.dot(current,current)
        a = np.dot(target,current)/np.dot(current,current)
        
        alpha = np.arctan(b/a)#Optimal angle of rotation is found.
        T = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        s =np.sqrt(np.pow(a,2) + np.pow(b,2))
        xFin = np.dot(s*T,xIni)
        error = np.linalg.norm(xFin-target)
        
        #Calculating the rotation matrix
        A[i,::2] = np.dot(T,current)[0,:];#Rotating the matrix X coordinate
        A[i,1::2] = np.dot(T,current)[1,:];#Rotating the matrix Y coordinate
        plt.plot(A[i,::2],A[i,1::2])
        plt.show()
    
    return
    

if __name__ == '__main__':
    reader = np.zeros([112,80])
    i=0;
    directory = "_Data/Landmarks/original/"
    for filename in fnmatch.filter(os.listdir(directory),'*.txt'):
        reader[i,:] = np.loadtxt(open(directory+filename,"rb"),delimiter=",",skiprows=0)
        reader[i,::2]  = reader[i,::2]-np.mean(reader[i,::2]);#Zero-mean of the X axis
        reader[i,1::2] = reader[i,1::2]-np.mean(reader[i,1::2]);#Zero-mean of the Y axis
        i+=1;
    #print reader[::8,:]
    shape, A,error = rescale(reader[::8,:]);
    [eigVals, eigVecs, mean] = PCA(A,0.98);
    result = Matching(target,eigVals,eigVecs,mean);
    
    plt.show()
    plt.plot(reader[0,::2],reader[0,1::2])
    plt.plot(shape[::2],shape[1::2])
