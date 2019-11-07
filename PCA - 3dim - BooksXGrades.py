# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 20:54:00 2019

@author: Victor Biazon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_table('Books_attend_grade.txt', decimal  = ",")

dataset = np.asarray(data)


def Covariance(data):
    Cov = np.zeros((len(data[0]),len(data[0])), dtype = float)
    for i in range(0, len(data[0])):
        for j in range(i, len(data[0])):
            sum = 0
            for k in range(0, len(data)):
                sum +=  data[k,i] * data[k,j]
            Cov[i,j] = sum/(len(data)-1)
            Cov[j,i] = Cov[i,j]
    return Cov


def EigenValues2Dim(CovMat): #calcula os autovalores de uma matrix de covariancia 2x2
    a = 1
    b = -(CovMat[0,0] + CovMat[1,1])
    c = CovMat[0,0] * CovMat[1,1] - CovMat[0,1] * CovMat[1,0]
    x1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
    x2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
    return x1, x2

def EigenValues3Dim(CovMat): #calcula os autovalores de uma matrix de covariancia 3x3
#       [(a-λ) b c][(a-λ) b]
#   det [d (e-λ) f][d (e-λ)] = (a-λ)(e-λ)(i-λ) + b*f*g + c*d*h - (g*(e-λ)*c + h*f*(a-λ) + (i-λ)*d*b) = λ³ + (a + e + i)λ² + (a*e + a*i + e*i)λ + (a*e*i)
#       [g h (i-λ)][ g   h ]                                                                         = -( -(g*c + h*f + d*b)λ + (g*e*c + h*f*a + i*d*b))
    a = 1
    b = -(CovMat[0,0] + CovMat[1,1])
    c = CovMat[0,0] * CovMat[1,1] - CovMat[0,1] * CovMat[1,0]
    x1 = (-b + math.sqrt(b**2 - 4*a*c))/(2*a)
    x2 = (-b - math.sqrt(b**2 - 4*a*c))/(2*a)
    x3 = x2
    return x1, x2, x3

def EigenVector2Dim(CovMat, EigVal): #calcula os autovetores de uma matriz de covariancia 2x2
#    [a b][x] = λ * [x]  --->>>   ax + by = λx      --->>>   (a - λ)x + by = 0                sendo x e y os termos do autovetor
#    [c d][y]       [y]           cx + dy = λy               cx + (d - λ)y = 0
    EigVect = np.zeros((2,2), float)
    x = 1
    for i in range(0,2):
        y = (CovMat[0,0] - EigVal[i])/(- CovMat[0,1]) * x
        EigVect[0,i] = x
        EigVect[1,i] = y
    return EigVect

def EigenVector3Dim(CovMat, EigVal): #calcula os autovetores de uma matriz de covariancia 3x3
#    [a b][x] = λ * [x]  --->>>   ax + by = λx      --->>>   (a - λ)x + by = 0                sendo x e y os termos do autovetor
#    [c d][y]       [y]           cx + dy = λy               cx + (d - λ)y = 0
    EigVect = np.zeros((2,2), float)
    x = 1
    for i in range(0,2):
        y = (CovMat[0,0] - EigVal[i])/(- CovMat[0,1]) * x
        EigVect[0,i] = x
        EigVect[1,i] = y
    return EigVect

def OrganizeEigen(EigenValue, EigVect):
    MatEigen = np.vstack((EigenValue,EigVect)) #junta os Eigenvalues na primeira linha com seus respectivos EigenVectors abaixo deles na coluna
    MatEigen = np.sort(MatEigen) #Ordena do menor pra maior pela linha de EigenValues
    MatEigen = MatEigen[:,::-1] #Inverte a ordem para o primeiro ser o maior EigenValue
    return MatEigen
  

def PCA(data):
    
    #extrair media dos dados
    Mean = np.mean(data) 
    data = np.asarray(data, dtype = float)
    M_data = np.copy(data)
    M_data[:,0] = data[:,0] - Mean.BOOKS
    M_data[:,1] = data[:,1] - Mean.ATTEND
    M_data[:,2] = data[:,2] - Mean.GRADE

    Cov = Covariance(M_data) #calcula matriz de covariancia
    EigV = np.linalg.eig(Cov) #calcula eigenvalues e eigenvectors
#    EigV = np.sort(EigV, axis = 0)
#    EigenValue = EigenValues3Dim(Cov)
#    EigVect = EigenVector3Dim(Cov, EigenValue)
    MatEigen = OrganizeEigen(EigV[0], EigV[1])
    FeatureVector = MatEigen[1:,:]
    FinalData = np.matmul(np.transpose(FeatureVector), np.transpose(M_data))
 
    M_data_pred_x = np.linspace(-3,3,20) * FeatureVector[0,0] # gera as retas das PCS para plotar no grafico
    M_data_pred_y = np.linspace(-10,10,20)  * FeatureVector[1,0]
    M_data_pred_z = np.linspace(-30,30,20)  * FeatureVector[2,0]
    M_data_pred_x2 = np.linspace(-3,3,20) * FeatureVector[0,1]
    M_data_pred_y2 = np.linspace(-10,10,20)  * FeatureVector[1,1]
    M_data_pred_z2 = np.linspace(-30,30,20)  * FeatureVector[2,1]
    M_data_pred_x3 = np.linspace(-3,3,20) * FeatureVector[0,2]
    M_data_pred_y3 = np.linspace(-10,10,20)  * FeatureVector[1,2]
    M_data_pred_z3 = np.linspace(-30,30,20)  * FeatureVector[2,2]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(M_data[:,0], M_data[:,1], M_data[:,2], zdir='z', s=20, c=None, depthshade=True)
    ax.plot(M_data_pred_x, M_data_pred_y, M_data_pred_z)
    ax.plot(M_data_pred_x2, M_data_pred_y2, M_data_pred_z2)
    ax.plot(M_data_pred_x3, M_data_pred_y3, M_data_pred_z3)
    
    

PCA(data)



