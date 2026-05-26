import numpy as np
import torch
import scipy as sp
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import pandas as pd
import time




def two_sol(r1,t1,r2,t2,ep1,ep2):
    r1x = r1[:,0]
    r1y = r1[:,1]
    r1z = r1[:,2]
    r2x = r2[:,0]
    r2y = r2[:,1]
    r2z = r2[:,2]
    r1x2 = r1x*r1x
    r1y2 = r1y*r1y
    r1z2 = r1z*r1z
    r2x2 = r2x*r2x
    r2y2 = r2y*r2y
    r2z2 = r2z*r2z
    ep12 = ep1*ep1
    ep22 = ep2*ep2
    t12 = t1*t1
    t22 = t2*t2

    f1 = (- ep12*r2x2 - ep12*r2y2 - ep12*r2z2 + 2*ep1*ep2*r1x*r2x + 2*ep1*ep2*r1y*r2y + 2*ep1*ep2*r1z*r2z + 2*ep1*r1x*r2x*t2 - 2*ep1*r2x2*t1 + 2*ep1*r1y*r2y*t2 - 2*ep1*r2y2*t1 + 2*ep1*r1z*r2z*t2 - 2*ep1*r2z2*t1 - ep22*r1x2 - ep22*r1y2 - ep22*r1z2 - 2*ep2*r1x2*t2 + 2*ep2*r1x*r2x*t1 - 2*ep2*r1y2*t2 + 2*ep2*r1y*r2y*t1 - 2*ep2*r1z2*t2 + 2*ep2*r1z*r2z*t1 + r1x2*r2y2 + r1x2*r2z2 - r1x2*t22 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + 2*r1x*r2x*t1*t2 + r2x2*r1y2 + r2x2*r1z2 - r2x2*t12 + r1y2*r2z2 - r1y2*t22 - 2*r1y*r2y*r1z*r2z + 2*r1y*r2y*t1*t2 + r2y2*r1z2 - r2y2*t12 - r1z2*t22 + 2*r1z*r2z*t1*t2 - r2z2*t12).abs()
    sqrt_f1 = torch.sqrt(f1)
    denom_xy = r1x*r2y - r2x*r1y
    denom_xyz = r1x2*r2y2 + r1x2*r2z2 - 2*r1x*r2x*r1y*r2y - 2*r1x*r2x*r1z*r2z + r2x2*r1y2 + r2x2*r1z2 + r1y2*r2z2 - 2*r1y*r2y*r1z*r2z + r2y2*r1z2
      
    q11 = (ep1*r2y - ep2*r1y - r1y*t2 + r2y*t1)/denom_xy + ((r1y*r2z - r2y*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 + r1x*r2y*sqrt_f1 - r2x*r1y*sqrt_f1 - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/(denom_xy*denom_xyz)
    q12 = (ep1*r2y - ep2*r1y - r1y*t2 + r2y*t1)/denom_xy + ((r1y*r2z - r2y*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 - r1x*r2y*sqrt_f1 + r2x*r1y*sqrt_f1 - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/(denom_xy*denom_xyz)
    q21 = - (ep1*r2x - ep2*r1x - r1x*t2 + r2x*t1)/denom_xy - ((r1x*r2z - r2x*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 + r1x*r2y*sqrt_f1 - r2x*r1y*sqrt_f1 - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/(denom_xy*denom_xyz)
    q22 = - (ep1*r2x - ep2*r1x - r1x*t2 + r2x*t1)/denom_xy - ((r1x*r2z - r2x*r1z)*(ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 - r1x*r2y*sqrt_f1 + r2x*r1y*sqrt_f1 - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1))/(denom_xy*denom_xyz)
    q31 = (ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 + r1x*r2y*sqrt_f1 - r2x*r1y*sqrt_f1 - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1)/denom_xyz
    q32 = (ep1*r2x2*r1z + ep2*r1x2*r2z + ep1*r2y2*r1z + ep2*r1y2*r2z + r2x2*r1z*t1 + r1x2*r2z*t2 + r2y2*r1z*t1 + r1y2*r2z*t2 - r1x*r2y*sqrt_f1 + r2x*r1y*sqrt_f1 - ep1*r1x*r2x*r2z - ep2*r1x*r2x*r1z - ep1*r1y*r2y*r2z - ep2*r1y*r2y*r1z - r1x*r2x*r1z*t2 - r1x*r2x*r2z*t1 - r1y*r2y*r1z*t2 - r1y*r2y*r2z*t1)/denom_xyz
    sol1 = torch.stack([q11,q21,q31],1)
    sol2 = torch.stack([q12,q22,q32],1)
 
    return sol1,sol2

    
  

def two_sol4(r1,t1,r2,t2,ep):
    sol1a,sol2a =  two_sol(r1,t1,r2,t2,ep,ep)
    sol1b,sol2b =  two_sol(r1,t1,r2,t2,-ep,ep)
    sol1c,sol2c =  two_sol(r1,t1,r2,t2,ep,-ep)
    sol1d,sol2d =  two_sol(r1,t1,r2,t2,-ep,-ep)
    sol = torch.vstack([sol1a, sol2a,sol1b, sol2b,sol1c, sol2c,sol1d, sol2d])
    return sol
    

def least_square(delta_r,tau):
    A = torch.stack([(delta_r[i,:].unsqueeze(0).T @ delta_r[i,:].unsqueeze(0)) for i in range(delta_r.shape[0])]).sum(0)
    b = (tau*delta_r).sum(0)
    a11 = A[0,0]
    a12 = A[0,1]
    a13 = A[0,2]
    a21 = A[1,0]
    a22 = A[1,1]
    a23 = A[1,2]
    a31 = A[2,0]
    a32 = A[2,1]
    a33 = A[2,2]
    b1 = b[0]
    b2 = b[1]
    b3 = b[2]
    a11_2 = a11*a11
    a12_2 = a12*a12
    a13_2 = a13*a13
    a21_2 = a21*a21
    a22_2 = a22*a22
    a23_2 = a23*a23
    a31_2 = a31*a31
    a32_2 = a32*a32
    a33_2 = a33*a33
    b1_2 = b1*b1
    b2_2 = b2*b2
    b3_2 = b3*b3
    p2 = a11_2*a22_2 - 2*a11_2*a22*a23 - 2*a11_2*a22*a32 + 2*a11_2*a22*a33 + a11_2*a23_2 + 2*a11_2*a23*a32 - 2*a11_2*a23*a33 + a11_2*a32_2 - 2*a11_2*a32*a33 + a11_2*a33_2 - 2*a11_2*b2_2 + 4*a11_2*b2*b3 - 2*a11_2*b3_2 - 2*a11*a12*a21*a22 + 2*a11*a12*a21*a23 + 2*a11*a12*a21*a32 - 2*a11*a12*a21*a33 + 2*a11*a12*a22*a23 + 2*a11*a12*a22*a31 - 2*a11*a12*a22*a33 - 2*a11*a12*a23_2 - 2*a11*a12*a23*a31 - 2*a11*a12*a23*a32 + 4*a11*a12*a23*a33 - 2*a11*a12*a31*a32 + 2*a11*a12*a31*a33 + 2*a11*a12*a32*a33 - 2*a11*a12*a33_2 + 2*a11*a12*b2_2 - 4*a11*a12*b2*b3 + 2*a11*a12*b3_2 + 2*a11*a13*a21*a22 - 2*a11*a13*a21*a23 - 2*a11*a13*a21*a32 + 2*a11*a13*a21*a33 - 2*a11*a13*a22_2 + 2*a11*a13*a22*a23 - 2*a11*a13*a22*a31 + 4*a11*a13*a22*a32 - 2*a11*a13*a22*a33 + 2*a11*a13*a23*a31 - 2*a11*a13*a23*a32 + 2*a11*a13*a31*a32 - 2*a11*a13*a31*a33 - 2*a11*a13*a32_2 + 2*a11*a13*a32*a33 + 2*a11*a13*b2_2 - 4*a11*a13*b2*b3 + 2*a11*a13*b3_2 + 2*a11*a21*a22*a32 - 2*a11*a21*a22*a33 - 2*a11*a21*a23*a32 + 2*a11*a21*a23*a33 - 2*a11*a21*a32_2 + 4*a11*a21*a32*a33 - 2*a11*a21*a33_2 + 4*a11*a21*b1*b2 - 4*a11*a21*b1*b3 - 4*a11*a21*b2*b3 + 4*a11*a21*b3_2 - 2*a11*a22_2*a31 + 2*a11*a22_2*a33 + 4*a11*a22*a23*a31 - 2*a11*a22*a23*a32 - 2*a11*a22*a23*a33 + 2*a11*a22*a31*a32 - 2*a11*a22*a31*a33 - 2*a11*a22*a32*a33 + 2*a11*a22*a33_2 - 2*a11*a22*b1*b2 + 2*a11*a22*b1*b3 + 2*a11*a22*b2*b3 - 2*a11*a22*b3_2 - 2*a11*a23_2*a31 + 2*a11*a23_2*a32 - 2*a11*a23*a31*a32 + 2*a11*a23*a31*a33 + 2*a11*a23*a32_2 - 2*a11*a23*a32*a33 - 2*a11*a23*b1*b2 + 2*a11*a23*b1*b3 + 2*a11*a23*b2*b3 - 2*a11*a23*b3_2 - 4*a11*a31*b1*b2 + 4*a11*a31*b1*b3 + 4*a11*a31*b2_2 - 4*a11*a31*b2*b3 + 2*a11*a32*b1*b2 - 2*a11*a32*b1*b3 - 2*a11*a32*b2_2 + 2*a11*a32*b2*b3 + 2*a11*a33*b1*b2 - 2*a11*a33*b1*b3 - 2*a11*a33*b2_2 + 2*a11*a33*b2*b3 + a12_2*a21_2 - 2*a12_2*a21*a23 - 2*a12_2*a21*a31 + 2*a12_2*a21*a33 + a12_2*a23_2 + 2*a12_2*a23*a31 - 2*a12_2*a23*a33 + a12_2*a31_2 - 2*a12_2*a31*a33 + a12_2*a33_2 - 2*a12_2*b2_2 + 4*a12_2*b2*b3 - 2*a12_2*b3_2 - 2*a12*a13*a21_2 + 2*a12*a13*a21*a22 + 2*a12*a13*a21*a23 + 4*a12*a13*a21*a31 - 2*a12*a13*a21*a32 - 2*a12*a13*a21*a33 - 2*a12*a13*a22*a23 - 2*a12*a13*a22*a31 + 2*a12*a13*a22*a33 - 2*a12*a13*a23*a31 + 2*a12*a13*a23*a32 - 2*a12*a13*a31_2 + 2*a12*a13*a31*a32 + 2*a12*a13*a31*a33 - 2*a12*a13*a32*a33 + 2*a12*a13*b2_2 - 4*a12*a13*b2*b3 + 2*a12*a13*b3_2 - 2*a12*a21_2*a32 + 2*a12*a21_2*a33 + 2*a12*a21*a22*a31 - 2*a12*a21*a22*a33 - 2*a12*a21*a23*a31 + 4*a12*a21*a23*a32 - 2*a12*a21*a23*a33 + 2*a12*a21*a31*a32 - 2*a12*a21*a31*a33 - 2*a12*a21*a32*a33 + 2*a12*a21*a33_2 - 2*a12*a21*b1*b2 + 2*a12*a21*b1*b3 + 2*a12*a21*b2*b3 - 2*a12*a21*b3_2 - 2*a12*a22*a23*a31 + 2*a12*a22*a23*a33 - 2*a12*a22*a31_2 + 4*a12*a22*a31*a33 - 2*a12*a22*a33_2 + 4*a12*a22*b1*b2 - 4*a12*a22*b1*b3 - 4*a12*a22*b2*b3 + 4*a12*a22*b3_2 + 2*a12*a23_2*a31 - 2*a12*a23_2*a32 + 2*a12*a23*a31_2 - 2*a12*a23*a31*a32 - 2*a12*a23*a31*a33 + 2*a12*a23*a32*a33 - 2*a12*a23*b1*b2 + 2*a12*a23*b1*b3 + 2*a12*a23*b2*b3 - 2*a12*a23*b3_2 + 2*a12*a31*b1*b2 - 2*a12*a31*b1*b3 - 2*a12*a31*b2_2 + 2*a12*a31*b2*b3 - 4*a12*a32*b1*b2 + 4*a12*a32*b1*b3 + 4*a12*a32*b2_2 - 4*a12*a32*b2*b3 + 2*a12*a33*b1*b2 - 2*a12*a33*b1*b3 - 2*a12*a33*b2_2 + 2*a12*a33*b2*b3 + a13_2*a21_2 - 2*a13_2*a21*a22 - 2*a13_2*a21*a31 + 2*a13_2*a21*a32 + a13_2*a22_2 + 2*a13_2*a22*a31 - 2*a13_2*a22*a32 + a13_2*a31_2 - 2*a13_2*a31*a32 + a13_2*a32_2 - 2*a13_2*b2_2 + 4*a13_2*b2*b3 - 2*a13_2*b3_2 + 2*a13*a21_2*a32 - 2*a13*a21_2*a33 - 2*a13*a21*a22*a31 - 2*a13*a21*a22*a32 + 4*a13*a21*a22*a33 + 2*a13*a21*a23*a31 - 2*a13*a21*a23*a32 - 2*a13*a21*a31*a32 + 2*a13*a21*a31*a33 + 2*a13*a21*a32_2 - 2*a13*a21*a32*a33 - 2*a13*a21*b1*b2 + 2*a13*a21*b1*b3 + 2*a13*a21*b2*b3 - 2*a13*a21*b3_2 + 2*a13*a22_2*a31 - 2*a13*a22_2*a33 - 2*a13*a22*a23*a31 + 2*a13*a22*a23*a32 + 2*a13*a22*a31_2 - 2*a13*a22*a31*a32 - 2*a13*a22*a31*a33 + 2*a13*a22*a32*a33 - 2*a13*a22*b1*b2 + 2*a13*a22*b1*b3 + 2*a13*a22*b2*b3 - 2*a13*a22*b3_2 - 2*a13*a23*a31_2 + 4*a13*a23*a31*a32 - 2*a13*a23*a32_2 + 4*a13*a23*b1*b2 - 4*a13*a23*b1*b3 - 4*a13*a23*b2*b3 + 4*a13*a23*b3_2 + 2*a13*a31*b1*b2 - 2*a13*a31*b1*b3 - 2*a13*a31*b2_2 + 2*a13*a31*b2*b3 + 2*a13*a32*b1*b2 - 2*a13*a32*b1*b3 - 2*a13*a32*b2_2 + 2*a13*a32*b2*b3 - 4*a13*a33*b1*b2 + 4*a13*a33*b1*b3 + 4*a13*a33*b2_2 - 4*a13*a33*b2*b3 + a21_2*a32_2 - 2*a21_2*a32*a33 + a21_2*a33_2 - 2*a21_2*b1_2 + 4*a21_2*b1*b3 - 2*a21_2*b3_2 - 2*a21*a22*a31*a32 + 2*a21*a22*a31*a33 + 2*a21*a22*a32*a33 - 2*a21*a22*a33_2 + 2*a21*a22*b1_2 - 4*a21*a22*b1*b3 + 2*a21*a22*b3_2 + 2*a21*a23*a31*a32 - 2*a21*a23*a31*a33 - 2*a21*a23*a32_2 + 2*a21*a23*a32*a33 + 2*a21*a23*b1_2 - 4*a21*a23*b1*b3 + 2*a21*a23*b3_2 + 4*a21*a31*b1_2 - 4*a21*a31*b1*b2 - 4*a21*a31*b1*b3 + 4*a21*a31*b2*b3 - 2*a21*a32*b1_2 + 2*a21*a32*b1*b2 + 2*a21*a32*b1*b3 - 2*a21*a32*b2*b3 - 2*a21*a33*b1_2 + 2*a21*a33*b1*b2 + 2*a21*a33*b1*b3 - 2*a21*a33*b2*b3 + a22_2*a31_2 - 2*a22_2*a31*a33 + a22_2*a33_2 - 2*a22_2*b1_2 + 4*a22_2*b1*b3 - 2*a22_2*b3_2 - 2*a22*a23*a31_2 + 2*a22*a23*a31*a32 + 2*a22*a23*a31*a33 - 2*a22*a23*a32*a33 + 2*a22*a23*b1_2 - 4*a22*a23*b1*b3 + 2*a22*a23*b3_2 - 2*a22*a31*b1_2 + 2*a22*a31*b1*b2 + 2*a22*a31*b1*b3 - 2*a22*a31*b2*b3 + 4*a22*a32*b1_2 - 4*a22*a32*b1*b2 - 4*a22*a32*b1*b3 + 4*a22*a32*b2*b3 - 2*a22*a33*b1_2 + 2*a22*a33*b1*b2 + 2*a22*a33*b1*b3 - 2*a22*a33*b2*b3 + a23_2*a31_2 - 2*a23_2*a31*a32 + a23_2*a32_2 - 2*a23_2*b1_2 + 4*a23_2*b1*b3 - 2*a23_2*b3_2 - 2*a23*a31*b1_2 + 2*a23*a31*b1*b2 + 2*a23*a31*b1*b3 - 2*a23*a31*b2*b3 - 2*a23*a32*b1_2 + 2*a23*a32*b1*b2 + 2*a23*a32*b1*b3 - 2*a23*a32*b2*b3 + 4*a23*a33*b1_2 - 4*a23*a33*b1*b2 - 4*a23*a33*b1*b3 + 4*a23*a33*b2*b3 - 2*a31_2*b1_2 + 4*a31_2*b1*b2 - 2*a31_2*b2_2 + 2*a31*a32*b1_2 - 4*a31*a32*b1*b2 + 2*a31*a32*b2_2 + 2*a31*a33*b1_2 - 4*a31*a33*b1*b2 + 2*a31*a33*b2_2 - 2*a32_2*b1_2 + 4*a32_2*b1*b2 - 2*a32_2*b2_2 + 2*a32*a33*b1_2 - 4*a32*a33*b1*b2 + 2*a32*a33*b2_2 - 2*a33_2*b1_2 + 4*a33_2*b1*b2 - 2*a33_2*b2_2
    p1 = 2*a11_2*a22_2*a33 - 2*a11_2*a22*a23*a32 - 2*a11_2*a22*a23*a33 - 2*a11_2*a22*a32*a33 + 2*a11_2*a22*a33_2 + 2*a11_2*a22*b2*b3 - 2*a11_2*a22*b3_2 + 2*a11_2*a23_2*a32 + 2*a11_2*a23*a32_2 - 2*a11_2*a23*a32*a33 + 2*a11_2*a23*b2*b3 - 2*a11_2*a23*b3_2 - 2*a11_2*a32*b2_2 + 2*a11_2*a32*b2*b3 - 2*a11_2*a33*b2_2 + 2*a11_2*a33*b2*b3 - 4*a11*a12*a21*a22*a33 + 2*a11*a12*a21*a23*a32 + 2*a11*a12*a21*a23*a33 + 2*a11*a12*a21*a32*a33 - 2*a11*a12*a21*a33_2 - 2*a11*a12*a21*b2*b3 + 2*a11*a12*a21*b3_2 + 2*a11*a12*a22*a23*a31 + 2*a11*a12*a22*a23*a33 + 2*a11*a12*a22*a31*a33 - 2*a11*a12*a22*a33_2 - 2*a11*a12*a22*b2*b3 + 2*a11*a12*a22*b3_2 - 2*a11*a12*a23_2*a31 - 2*a11*a12*a23_2*a32 - 4*a11*a12*a23*a31*a32 + 2*a11*a12*a23*a31*a33 + 2*a11*a12*a23*a32*a33 + 2*a11*a12*a31*b2_2 - 2*a11*a12*a31*b2*b3 + 2*a11*a12*a32*b2_2 - 2*a11*a12*a32*b2*b3 + 2*a11*a13*a21*a22*a32 + 2*a11*a13*a21*a22*a33 - 4*a11*a13*a21*a23*a32 - 2*a11*a13*a21*a32_2 + 2*a11*a13*a21*a32*a33 - 2*a11*a13*a21*b2*b3 + 2*a11*a13*a21*b3_2 - 2*a11*a13*a22_2*a31 - 2*a11*a13*a22_2*a33 + 2*a11*a13*a22*a23*a31 + 2*a11*a13*a22*a23*a32 + 2*a11*a13*a22*a31*a32 - 4*a11*a13*a22*a31*a33 + 2*a11*a13*a22*a32*a33 + 2*a11*a13*a23*a31*a32 - 2*a11*a13*a23*a32_2 - 2*a11*a13*a23*b2*b3 + 2*a11*a13*a23*b3_2 + 2*a11*a13*a31*b2_2 - 2*a11*a13*a31*b2*b3 + 2*a11*a13*a33*b2_2 - 2*a11*a13*a33*b2*b3 + 2*a11*a21*a22*a32*a33 - 2*a11*a21*a22*a33_2 - 2*a11*a21*a22*b1*b3 + 2*a11*a21*a22*b3_2 - 2*a11*a21*a23*a32_2 + 2*a11*a21*a23*a32*a33 - 2*a11*a21*a23*b1*b3 + 2*a11*a21*a23*b3_2 + 4*a11*a21*a32*b1*b2 - 2*a11*a21*a32*b1*b3 - 2*a11*a21*a32*b2*b3 + 4*a11*a21*a33*b1*b2 - 2*a11*a21*a33*b1*b3 - 2*a11*a21*a33*b2*b3 - 2*a11*a22_2*a31*a33 + 2*a11*a22_2*a33_2 + 2*a11*a22_2*b1*b3 - 2*a11*a22_2*b3_2 + 2*a11*a22*a23*a31*a32 + 2*a11*a22*a23*a31*a33 - 4*a11*a22*a23*a32*a33 - 2*a11*a22*a31*b1*b2 + 4*a11*a22*a31*b1*b3 - 2*a11*a22*a31*b2*b3 - 2*a11*a22*a32*b1*b2 - 2*a11*a22*a32*b1*b3 + 4*a11*a22*a32*b2*b3 - 2*a11*a23_2*a31*a32 + 2*a11*a23_2*a32_2 + 2*a11*a23_2*b1*b3 - 2*a11*a23_2*b3_2 - 2*a11*a23*a31*b1*b2 + 4*a11*a23*a31*b1*b3 - 2*a11*a23*a31*b2*b3 - 2*a11*a23*a33*b1*b2 - 2*a11*a23*a33*b1*b3 + 4*a11*a23*a33*b2*b3 - 2*a11*a31*a32*b1*b2 + 2*a11*a31*a32*b2_2 - 2*a11*a31*a33*b1*b2 + 2*a11*a31*a33*b2_2 + 2*a11*a32_2*b1*b2 - 2*a11*a32_2*b2_2 + 2*a11*a33_2*b1*b2 - 2*a11*a33_2*b2_2 + 2*a12_2*a21_2*a33 - 2*a12_2*a21*a23*a31 - 2*a12_2*a21*a23*a33 - 2*a12_2*a21*a31*a33 + 2*a12_2*a21*a33_2 + 2*a12_2*a21*b2*b3 - 2*a12_2*a21*b3_2 + 2*a12_2*a23_2*a31 + 2*a12_2*a23*a31_2 - 2*a12_2*a23*a31*a33 + 2*a12_2*a23*b2*b3 - 2*a12_2*a23*b3_2 - 2*a12_2*a31*b2_2 + 2*a12_2*a31*b2*b3 - 2*a12_2*a33*b2_2 + 2*a12_2*a33*b2*b3 - 2*a12*a13*a21_2*a32 - 2*a12*a13*a21_2*a33 + 2*a12*a13*a21*a22*a31 + 2*a12*a13*a21*a22*a33 + 2*a12*a13*a21*a23*a31 + 2*a12*a13*a21*a23*a32 + 2*a12*a13*a21*a31*a32 + 2*a12*a13*a21*a31*a33 - 4*a12*a13*a21*a32*a33 - 4*a12*a13*a22*a23*a31 - 2*a12*a13*a22*a31_2 + 2*a12*a13*a22*a31*a33 - 2*a12*a13*a22*b2*b3 + 2*a12*a13*a22*b3_2 - 2*a12*a13*a23*a31_2 + 2*a12*a13*a23*a31*a32 - 2*a12*a13*a23*b2*b3 + 2*a12*a13*a23*b3_2 + 2*a12*a13*a32*b2_2 - 2*a12*a13*a32*b2*b3 + 2*a12*a13*a33*b2_2 - 2*a12*a13*a33*b2*b3 - 2*a12*a21_2*a32*a33 + 2*a12*a21_2*a33_2 + 2*a12*a21_2*b1*b3 - 2*a12*a21_2*b3_2 + 2*a12*a21*a22*a31*a33 - 2*a12*a21*a22*a33_2 - 2*a12*a21*a22*b1*b3 + 2*a12*a21*a22*b3_2 + 2*a12*a21*a23*a31*a32 - 4*a12*a21*a23*a31*a33 + 2*a12*a21*a23*a32*a33 - 2*a12*a21*a31*b1*b2 - 2*a12*a21*a31*b1*b3 + 4*a12*a21*a31*b2*b3 - 2*a12*a21*a32*b1*b2 + 4*a12*a21*a32*b1*b3 - 2*a12*a21*a32*b2*b3 - 2*a12*a22*a23*a31_2 + 2*a12*a22*a23*a31*a33 - 2*a12*a22*a23*b1*b3 + 2*a12*a22*a23*b3_2 + 4*a12*a22*a31*b1*b2 - 2*a12*a22*a31*b1*b3 - 2*a12*a22*a31*b2*b3 + 4*a12*a22*a33*b1*b2 - 2*a12*a22*a33*b1*b3 - 2*a12*a22*a33*b2*b3 + 2*a12*a23_2*a31_2 - 2*a12*a23_2*a31*a32 + 2*a12*a23_2*b1*b3 - 2*a12*a23_2*b3_2 - 2*a12*a23*a32*b1*b2 + 4*a12*a23*a32*b1*b3 - 2*a12*a23*a32*b2*b3 - 2*a12*a23*a33*b1*b2 - 2*a12*a23*a33*b1*b3 + 4*a12*a23*a33*b2*b3 + 2*a12*a31_2*b1*b2 - 2*a12*a31_2*b2_2 - 2*a12*a31*a32*b1*b2 + 2*a12*a31*a32*b2_2 - 2*a12*a32*a33*b1*b2 + 2*a12*a32*a33*b2_2 + 2*a12*a33_2*b1*b2 - 2*a12*a33_2*b2_2 + 2*a13_2*a21_2*a32 - 2*a13_2*a21*a22*a31 - 2*a13_2*a21*a22*a32 - 2*a13_2*a21*a31*a32 + 2*a13_2*a21*a32_2 + 2*a13_2*a21*b2*b3 - 2*a13_2*a21*b3_2 + 2*a13_2*a22_2*a31 + 2*a13_2*a22*a31_2 - 2*a13_2*a22*a31*a32 + 2*a13_2*a22*b2*b3 - 2*a13_2*a22*b3_2 - 2*a13_2*a31*b2_2 + 2*a13_2*a31*b2*b3 - 2*a13_2*a32*b2_2 + 2*a13_2*a32*b2*b3 + 2*a13*a21_2*a32_2 - 2*a13*a21_2*a32*a33 + 2*a13*a21_2*b1*b3 - 2*a13*a21_2*b3_2 - 4*a13*a21*a22*a31*a32 + 2*a13*a21*a22*a31*a33 + 2*a13*a21*a22*a32*a33 + 2*a13*a21*a23*a31*a32 - 2*a13*a21*a23*a32_2 - 2*a13*a21*a23*b1*b3 + 2*a13*a21*a23*b3_2 - 2*a13*a21*a31*b1*b2 - 2*a13*a21*a31*b1*b3 + 4*a13*a21*a31*b2*b3 - 2*a13*a21*a33*b1*b2 + 4*a13*a21*a33*b1*b3 - 2*a13*a21*a33*b2*b3 + 2*a13*a22_2*a31_2 - 2*a13*a22_2*a31*a33 + 2*a13*a22_2*b1*b3 - 2*a13*a22_2*b3_2 - 2*a13*a22*a23*a31_2 + 2*a13*a22*a23*a31*a32 - 2*a13*a22*a23*b1*b3 + 2*a13*a22*a23*b3_2 - 2*a13*a22*a32*b1*b2 - 2*a13*a22*a32*b1*b3 + 4*a13*a22*a32*b2*b3 - 2*a13*a22*a33*b1*b2 + 4*a13*a22*a33*b1*b3 - 2*a13*a22*a33*b2*b3 + 4*a13*a23*a31*b1*b2 - 2*a13*a23*a31*b1*b3 - 2*a13*a23*a31*b2*b3 + 4*a13*a23*a32*b1*b2 - 2*a13*a23*a32*b1*b3 - 2*a13*a23*a32*b2*b3 + 2*a13*a31_2*b1*b2 - 2*a13*a31_2*b2_2 - 2*a13*a31*a33*b1*b2 + 2*a13*a31*a33*b2_2 + 2*a13*a32_2*b1*b2 - 2*a13*a32_2*b2_2 - 2*a13*a32*a33*b1*b2 + 2*a13*a32*a33*b2_2 - 2*a21_2*a32*b1_2 + 2*a21_2*a32*b1*b3 - 2*a21_2*a33*b1_2 + 2*a21_2*a33*b1*b3 + 2*a21*a22*a31*b1_2 - 2*a21*a22*a31*b1*b3 + 2*a21*a22*a32*b1_2 - 2*a21*a22*a32*b1*b3 + 2*a21*a23*a31*b1_2 - 2*a21*a23*a31*b1*b3 + 2*a21*a23*a33*b1_2 - 2*a21*a23*a33*b1*b3 + 2*a21*a31*a32*b1_2 - 2*a21*a31*a32*b1*b2 + 2*a21*a31*a33*b1_2 - 2*a21*a31*a33*b1*b2 - 2*a21*a32_2*b1_2 + 2*a21*a32_2*b1*b2 - 2*a21*a33_2*b1_2 + 2*a21*a33_2*b1*b2 - 2*a22_2*a31*b1_2 + 2*a22_2*a31*b1*b3 - 2*a22_2*a33*b1_2 + 2*a22_2*a33*b1*b3 + 2*a22*a23*a32*b1_2 - 2*a22*a23*a32*b1*b3 + 2*a22*a23*a33*b1_2 - 2*a22*a23*a33*b1*b3 - 2*a22*a31_2*b1_2 + 2*a22*a31_2*b1*b2 + 2*a22*a31*a32*b1_2 - 2*a22*a31*a32*b1*b2 + 2*a22*a32*a33*b1_2 - 2*a22*a32*a33*b1*b2 - 2*a22*a33_2*b1_2 + 2*a22*a33_2*b1*b2 - 2*a23_2*a31*b1_2 + 2*a23_2*a31*b1*b3 - 2*a23_2*a32*b1_2 + 2*a23_2*a32*b1*b3 - 2*a23*a31_2*b1_2 + 2*a23*a31_2*b1*b2 + 2*a23*a31*a33*b1_2 - 2*a23*a31*a33*b1*b2 - 2*a23*a32_2*b1_2 + 2*a23*a32_2*b1*b2 + 2*a23*a32*a33*b1_2 - 2*a23*a32*a33*b1*b2
    p0 = a11_2*a22_2*a33_2 - a11_2*a22_2*b3_2 - 2*a11_2*a22*a23*a32*a33 + 2*a11_2*a22*a32*b2*b3 + a11_2*a23_2*a32_2 - a11_2*a23_2*b3_2 + 2*a11_2*a23*a33*b2*b3 - a11_2*a32_2*b2_2 - a11_2*a33_2*b2_2 - 2*a11*a12*a21*a22*a33_2 + 2*a11*a12*a21*a22*b3_2 + 2*a11*a12*a21*a23*a32*a33 - 2*a11*a12*a21*a32*b2*b3 + 2*a11*a12*a22*a23*a31*a33 - 2*a11*a12*a22*a31*b2*b3 - 2*a11*a12*a23_2*a31*a32 + 2*a11*a12*a31*a32*b2_2 + 2*a11*a13*a21*a22*a32*a33 - 2*a11*a13*a21*a23*a32_2 + 2*a11*a13*a21*a23*b3_2 - 2*a11*a13*a21*a33*b2*b3 - 2*a11*a13*a22_2*a31*a33 + 2*a11*a13*a22*a23*a31*a32 - 2*a11*a13*a23*a31*b2*b3 + 2*a11*a13*a31*a33*b2_2 - 2*a11*a21*a22*a32*b1*b3 - 2*a11*a21*a23*a33*b1*b3 + 2*a11*a21*a32_2*b1*b2 + 2*a11*a21*a33_2*b1*b2 + 2*a11*a22_2*a31*b1*b3 - 2*a11*a22*a31*a32*b1*b2 + 2*a11*a23_2*a31*b1*b3 - 2*a11*a23*a31*a33*b1*b2 + a12_2*a21_2*a33_2 - a12_2*a21_2*b3_2 - 2*a12_2*a21*a23*a31*a33 + 2*a12_2*a21*a31*b2*b3 + a12_2*a23_2*a31_2 - a12_2*a23_2*b3_2 + 2*a12_2*a23*a33*b2*b3 - a12_2*a31_2*b2_2 - a12_2*a33_2*b2_2 - 2*a12*a13*a21_2*a32*a33 + 2*a12*a13*a21*a22*a31*a33 + 2*a12*a13*a21*a23*a31*a32 - 2*a12*a13*a22*a23*a31_2 + 2*a12*a13*a22*a23*b3_2 - 2*a12*a13*a22*a33*b2*b3 - 2*a12*a13*a23*a32*b2*b3 + 2*a12*a13*a32*a33*b2_2 + 2*a12*a21_2*a32*b1*b3 - 2*a12*a21*a22*a31*b1*b3 - 2*a12*a21*a31*a32*b1*b2 - 2*a12*a22*a23*a33*b1*b3 + 2*a12*a22*a31_2*b1*b2 + 2*a12*a22*a33_2*b1*b2 + 2*a12*a23_2*a32*b1*b3 - 2*a12*a23*a32*a33*b1*b2 + a13_2*a21_2*a32_2 - a13_2*a21_2*b3_2 - 2*a13_2*a21*a22*a31*a32 + 2*a13_2*a21*a31*b2*b3 + a13_2*a22_2*a31_2 - a13_2*a22_2*b3_2 + 2*a13_2*a22*a32*b2*b3 - a13_2*a31_2*b2_2 - a13_2*a32_2*b2_2 + 2*a13*a21_2*a33*b1*b3 - 2*a13*a21*a23*a31*b1*b3 - 2*a13*a21*a31*a33*b1*b2 + 2*a13*a22_2*a33*b1*b3 - 2*a13*a22*a23*a32*b1*b3 - 2*a13*a22*a32*a33*b1*b2 + 2*a13*a23*a31_2*b1*b2 + 2*a13*a23*a32_2*b1*b2 - a21_2*a32_2*b1_2 - a21_2*a33_2*b1_2 + 2*a21*a22*a31*a32*b1_2 + 2*a21*a23*a31*a33*b1_2 - a22_2*a31_2*b1_2 - a22_2*a33_2*b1_2 + 2*a22*a23*a32*a33*b1_2 - a23_2*a31_2*b1_2 - a23_2*a32_2*b1_2
    if (p1*p1-4*p0*p2) >= 0:
        lam1 = -(p1 - np.sqrt(p1*p1 - 4*p0*p2))/(2*p2)
        lam2 = -(p1 + np.sqrt(p1*p1 - 4*p0*p2))/(2*p2)
        v1 = torch.linalg.solve(A+lam1,b)
        v2 = torch.linalg.solve(A+lam2,b)
        res1 = delta_r@v1.unsqueeze(1)-tau
        res2 = delta_r@v2.unsqueeze(1)-tau
        min1 = np.linalg.norm(res1)
        min2 = np.linalg.norm(res2)
        return (v1,min1) if min1 < min2 else (v2,min2)
    else:
        return(0,-1)
        
  

@torch.inference_mode()
def find_inlier_max_doa(recdiff, amed, ep = 0.5):
    vv_est = []
    vv_ls_est = []

    sz = recdiff.shape[0]
    pair_i, pair_j = torch.triu_indices(sz, sz, offset=1, device=recdiff.device)
    r1 = recdiff[pair_j, :]
    r2 = recdiff[pair_i, :]



    for k in range(amed.shape[0]):
        t1 = amed[k, pair_j]
        t2 = amed[k, pair_i]
        sol = two_sol4(r1,t1,r2,t2,ep)
        resok = ((sol@recdiff.T-amed[k,:]).abs()<=ep).sum(1)
        besti = resok.argmax()
        vv = sol[besti,:]     
        vv_est.append(vv)                
        inliers = (vv@recdiff.T-amed[k,:]).abs()<=ep
        rd_in = recdiff[inliers,:]
        tau_in = amed[k,inliers]

        vv_ls,mini = least_square(rd_in,tau_in.unsqueeze(1))
        if mini>0:
            vv_ls_est.append(vv_ls)
        else:
            vv_ls_est.append(vv)

    vv_est = torch.stack(vv_est)
    vv_ls_est = torch.stack(vv_ls_est)
    return vv_est, vv_ls_est




@torch.inference_mode()
def find_inlier_max_doa_pos_phi(recdiff, amed, ep = 0.5):
    vv_est = []
    vv_ls_est = []

    sz = recdiff.shape[0]
    pair_i, pair_j = torch.triu_indices(sz, sz, offset=1, device=recdiff.device)
    r1 = recdiff[pair_j, :]
    r2 = recdiff[pair_i, :]



    for k in range(amed.shape[0]):
        t1 = amed[k, pair_j]
        t2 = amed[k, pair_i]
        sol = two_sol4(r1,t1,r2,t2,ep)
        resok = ((sol@recdiff.T-amed[k,:]).abs()<=ep).sum(1)
        c = sol[:,2]
        resok[c<0] = 0

        besti = resok.argmax()
        vv = sol[besti,:]     
        vv_est.append(vv)                
        inliers = (vv@recdiff.T-amed[k,:]).abs()<=ep
        rd_in = recdiff[inliers,:]
        tau_in = amed[k,inliers]

        vv_ls,mini = least_square(rd_in,tau_in.unsqueeze(1))
        if mini>0:
            vv_ls_est.append(vv_ls)
        else:
            vv_ls_est.append(vv)

    vv_est = torch.stack(vv_est)
    vv_ls_est = torch.stack(vv_ls_est)
    return vv_est, vv_ls_est


if __name__ == "__main__":
    print("Load data")
    experiment_folder = "../data/ljungbyhed_audio_dataset/01_flower/"
    mic_pos_file = "../new_receivers_positions.csv"
    df = pd.read_csv(mic_pos_file)
    rec = df.to_numpy()
    rec = rec - rec.mean(0)
    dx = rec[11]-rec[5]
    dy = rec[9]-rec[3]
    dy = dy-(dy.T@dx)*dx/(dx.T@dx)
    dz = np.cross(dx, dy)
    dx = dx / np.linalg.norm(dx)
    dy = dy / np.linalg.norm(dy)
    dz = dz / np.linalg.norm(dz)
    R = np.stack([dx, dy, dz]).T
    rec = torch.tensor((R.T @ rec.T).T,dtype=torch.float32)
    fs,slong = sp.io.wavfile.read(experiment_folder + "sound.wav")
    slong = slong.T
    print("Preprocess data")
    diffs = []
    duration = 0.1
    for start_time in torch.arange(0,slong.shape[1]/fs,duration):
        stime = time.process_time() # starting processtime
        hz_limit_low = 200 #high pass filter
        win_len = int(duration*fs)
        s = torch.tensor(slong[:,int(start_time*fs):int((start_time+duration)*fs+1)])
        s = s[:,1:] - s[:,:-1]
        sfft = np.fft.fft(s)
        hz_lims_low = 4000 
        hz_lims_high = 6000 
        component_lim_low = int(hz_lims_low*duration)
        component_lim_high = int(hz_lims_high*duration)
        sfft[:,:component_lim_low+1] = 0
        sfft[:,sfft.shape[-1] - component_lim_low:] = 0
        sfft[:,component_lim_high+1:sfft.shape[-1] - component_lim_high] = 0
        local_diff = []
        for i in range(12):
            for j in range(i+1,12):
                temp = sfft[i] * sfft[j].conj()
                temp = temp/(np.abs(temp) + 1e-5)
                re = np.fft.fftshift(np.fft.ifft(temp).real)
                local_diff.append(343*(re.argmax() - duration*fs/2)/fs) 
        diffs.append(local_diff)
        etime = time.process_time() # ending processtime
    print("median filter")
    a = torch.tensor(diffs)
    tmp = torch.tensor(local_diff)
    amed = []
    median_window = 30
    for i in range(a.shape[0]- median_window):
        amed.append(a[i:i+median_window].median(0).values)
    amed = torch.stack(amed)
    recdiff = torch.cat([torch.stack([(rec[j,:] - rec[i,:]) for j in range(i+1,12)]) for i in range(11)]).to(torch.float64)
    ep = 0.1
    print("Inlier maximization")
    vv_est, vv_ls_est = find_inlier_max_doa(recdiff, amed, ep)
