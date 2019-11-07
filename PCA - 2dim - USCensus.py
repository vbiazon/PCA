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

data = pd.read_table('USCensus2.txt', decimal  = ".")

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

def OrganizeEigen(EigenValue, EigVect):
    MatEigen = np.vstack((EigenValue,EigVect)) #junta os Eigenvalues na primeira linha com seus respectivos EigenVectors abaixo deles na coluna
    MatEigen = np.sort(MatEigen) #Ordena do menor pra maior pela linha de EigenValues
    MatEigen = MatEigen[:,::-1] #Inverte a ordem para o primeiro ser o maior EigenValue
    return MatEigen
  

def PCA(data):
    
    #extrair media dos dados
    Mean = np.mean(data)  #calcula medias dos valores das colunas do dataset
    data = np.asarray(data)
    M_data = np.copy(data)
    M_data[:,0] = data[:,0] - Mean.Year #retira media da primeira coluna
    M_data[:,1] = data[:,1] - Mean.Population #retira media da segunda coluna

    Cov = Covariance(M_data) #calcula covariancia do dataset
    EigenValue = EigenValues2Dim(Cov) #calcula autovalores
    EigVect = EigenVector2Dim(Cov, EigenValue) #calcula autovetores
    MatEigen = OrganizeEigen(EigenValue, EigVect) #organiza autovalores e vetores em ordem crescente de relevancia para analise
    FeatureVector = MatEigen[1:,:]
    FinalData = np.matmul(np.transpose(FeatureVector), np.transpose(M_data))
 
    M_data_pred_x = np.linspace(-60,60,20) * EigVect[0,0]
    M_data_pred_y = M_data_pred_x * EigVect[1,0]
    M_data_pred_x2 = np.linspace(-70,70,20) * EigVect[0,1]
    M_data_pred_y2 = M_data_pred_x2 * EigVect[1,1]

    
    plt.figure()
    plt.scatter(M_data[:,0], M_data[:,1], color = 'blue')
    plt.plot(M_data_pred_x,M_data_pred_y, color = 'green')
    plt.plot(M_data_pred_x2,M_data_pred_y2, color = 'green')    
    plt.title('Dados crescimento da população ao longo dos anos')
    plt.xlabel('Anos')
    plt.ylabel('População')
    plt.show()
    
#    plt.figure()
#    plt.scatter(M_data[:,0] + Mean.Year, M_data[:,1] + Mean.Population, color = 'blue')
#    plt.plot(M_data_pred_x  + Mean.Year, M_data_pred_y + Mean.Population, color = 'green')
#    plt.title('Dados crescimento da população ao longo dos anos')
#    plt.xlabel('Anos')
#    plt.ylabel('População')
#    plt.show()
    
    

PCA(data)



