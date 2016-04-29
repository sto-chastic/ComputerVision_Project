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
import scipy


def rescale(A):
    [c,r] = A.shape
    mean = A[0,:];#Assign the first image as the MEAN image
    error = 10;
    
    old = np.zeros((1,80))
    j=0
    while error>0.00001:
        scale = np.sqrt(np.sum(np.square(mean)))
        
        A = np.divide(A,scale)
        mean = np.divide(mean,scale)
        old[:] = mean[:];
        
        for i in range(c):#For for calculating the rotation of the matrices.
            #plt.plot(A[i,::2],A[i,1::2])
            #plt.plot(mean[::2],mean[1::2])
            
            s, a, T, A[i,:]= transform(A[i,:], mean);
            #plt.plot(A[i,::2],A[i,1::2])
            A[i,:] = project_tangent(A[i,:], mean); #Rotating the matrix X coordinate
            #plt.plot(A[i,::2],A[i,1::2])
            plt.show()
            
        mean[:] = np.mean(A,0)#Create a new mean matrix based on the mean of the rotated and scaled shapes.
        scale = np.sqrt(np.sum(np.square(mean)));
        mean[:] = mean /scale;
        error = np.linalg.norm(mean-old);
        print j
        print error
    return mean,A,error


    
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
        XIni = mean + np.dot(eigVecs,b)
        
        s, a, T, xFin = transform(XIni, target)
        xFin = project_tangent(xFin);
        
        error = np.linalg.norm(xFin-target)
        
        plt.plot(A[i,::2],A[i,1::2])
        plt.show()
        
        print error
        
    
    return

def project_tangent(A, T):
    '''
    Project onto tangent space
    @param A:               
    @param T:                            
    @return: s = scaling, alpha = angle, T = transformation matrix
    '''
    tangent = np.dot(A, T);
    A_new = A/tangent;
    return A_new


def transform(A, T):
    '''
    Calculate scaling and theta angle of image A to target T
    @param A:               
    @param T:                            
    @return: s = scaling, alpha = angle, T = transformation matrix
    '''
    Ax, Ay = split(A)
    Tx, Ty = split(T)

    b2 = (np.dot(Ax, Ty)-np.dot(Ay, Tx))/np.power(np.dot(A, A),2)
    a2 = np.dot(T, A)/np.power(np.dot(A, A),2)
        
    alpha = np.arctan(b2/a2) #Optimal angle of rotation is found.
    T = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
    s =np.sqrt(np.power(a2,2) + np.power(b2,2))
    
    result = np.dot(s*T, np.vstack((Ax,Ay)));
    plt.plot(result[0,:],result[1,:])
    new_A = merge(result);
    
    return s, alpha, T, new_A
    
def split (A):
    x = A[::2];#Divide in X coordinates
    y = A[1::2];#Divide in Y coordinates
    return x,y

def merge(XY):
    A = np.zeros((1,XY.shape[1]*2))
    A[0,::2] = XY[0, :] ;
    A[0,1::2] = XY[1, :];
    return A

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
    x = X - Xm
    x = x.T
    Xc = np.dot(x.T,x)
    [L,V] = LA.eig(Xc)
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
    
def model_learning(t_size,data):
    
    # split dataset in Target and Training set
    # Do it for all 
    [c,r] = data.shape
    target = np.zeros((t_size, 80))
    training = np.zeros((c-t_size, 80))
    for i in range(c/t_size):
        start = i*t_size
        stop =  i+t_size
        target[:,:]= data[start:stop, :]
        training[:,:] = data
        model, A, error = rescale(training);
        [eigVals, eigVecs, mean] = PCA(model,0.98);
        
        for i in range(t_size):
            result = Matching(target,eigVals,eigVecs,mean);
            print result

    return model, error
    
def preproc():
    img = cv2.imread('5DnwY.jpg', 0)
    # Number of rows and columns
    [cols, rows] = img.shape
    
    # Remove some columns from the beginning and end
    img = img[:, 59:cols-20]

    # Convert image to 0 to 1, then do log(1 + I)
    imgLog = np.log1p(np.array(img, dtype="float") / 255)
    
    # Create Gaussian mask of sigma = 10
    M = 2*rows + 1
    N = 2*cols + 1
    sigma = 10
    (X,Y) = np.meshgrid(np.linspace(0,N-1,N), np.linspace(0,M-1,M))
    centerX = np.ceil(N/2)
    centerY = np.ceil(M/2)
    gaussianNumerator = (X - centerX)**2 + (Y - centerY)**2
    
    # Low pass and high pass filters
    Hlow = np.exp(-gaussianNumerator / (2*sigma*sigma))
    Hhigh = 1 - Hlow
    
    # Move origin of filters so that it's at the top left corner to
    # match with the input image
    
    HlowShift = scipy.fftpack.ifftshift(Hlow.copy())
    HhighShift = scipy.fftpack.ifftshift(Hhigh.copy())
    
    # Filter the image and crop
    If = scipy.fftpack.fft2(imgLog.copy(), (M,N))
    Ioutlow = scipy.real(scipy.fftpack.ifft2(If.copy() * HlowShift, (M,N)))
    Iouthigh = scipy.real(scipy.fftpack.ifft2(If.copy() * HhighShift, (M,N)))
    
    # Set scaling factors and add
    gamma1 = 0.3
    gamma2 = 1.5
    Iout = gamma1*Ioutlow[0:rows,0:cols] + gamma2*Iouthigh[0:rows,0:cols]
    
    # Anti-log then rescale to [0,1]
    Ihmf = np.expm1(Iout)
    Ihmf = (Ihmf - np.min(Ihmf)) / (np.max(Ihmf) - np.min(Ihmf))
    Ihmf2 = np.array(255*Ihmf, dtype="uint8")
    
    # Threshold the image - Anything below intensity 65 gets set to white
    Ithresh = Ihmf2 < 65
    Ithresh = 255*Ithresh.astype("uint8")
    
    # Clear off the border.  Choose a border radius of 5 pixels
    Iclear = imclearborder(Ithresh, 5)
    
    # Eliminate regions that have areas below 120 pixels
    Iopen = bwareaopen(Iclear, 120)
    
    # Show all images
    cv2.imshow('Original Image', img)
    cv2.imshow('Homomorphic Filtered Result', Ihmf2)
    cv2.imshow('Thresholded Result', Ithresh)
    cv2.imshow('Opened Result', Iopen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    reader = np.zeros([112,80])
    i=0;
    directory = "C:\Users\David\Google Drive\KULeuven\Computer vision\Nieuwe map\\_Data/Landmarks/original/"
    for filename in fnmatch.filter(os.listdir(directory),'*.txt'):
        reader[i,:] = np.loadtxt(open(directory+filename,"rb"),delimiter=",",skiprows=0)
        reader[i,::2]  = reader[i,::2]-np.mean(reader[i,::2]);#Zero-mean of the X axis
        reader[i,1::2] = reader[i,1::2]-np.mean(reader[i,1::2]);#Zero-mean of the Y axis
        i+=1;
    #print reader[::8,:]
    #shape, A,error = rescale(reader[::8,:]);
    model_learning(8,reader)
    plt.show()
    plt.plot(reader[0,::2],reader[0,1::2])
    plt.plot(shape[::2],shape[1::2])
