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

data = pd.read_table('alpswater.txt', decimal  = ",") #importação dos dados dos datasets

dataset = np.asarray(data) #transforma dados em Array


def Covariance(data): #calcula a matriz de covariancia dos dados
    Cov = np.zeros((len(data[0]),len(data[0])), dtype = float) #inicializa uma matriz para armazenar os dados
    for i in range(0, len(data[0])): #calcula para cada relaçao de colunas as variancias de uma para a outra
        for j in range(i, len(data[0])):
            sum = 0
            for k in range(0, len(data)):
                sum +=  data[k,i] * data[k,j]
            Cov[i,j] = sum/(len(data)-1)
            Cov[j,i] = Cov[i,j] #como é uma matriz simétrica calcula apenas para [i,j] e replica para [j,i]
    return Cov


def EigenValues2Dim(CovMat): #calcula os autovalores de uma matrix de covariancia 2x2
    #como a matriz de covariancia é 2x2 neste caso, e acha os autovalores igualando o determinante  de (A -λI) a 0 e se acha as raizes por Bhaskara
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
    for i in range(0,2): #calcula os autovetores achando a relação entre X e Y pelo sistema linear
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
    

    Mean = np.mean(data) #calcula medias dos valores das colunas do dataset
    data = np.asarray(data)
    M_data = np.copy(data)
    M_data[:,0] = data[:,0] - Mean.BPt #retira media da primeira coluna
    M_data[:,1] = data[:,1] - Mean.Pressure #retira media da segunda coluna

    Cov = Covariance(M_data) #calcula covariancia do dataset
    EigenValue = EigenValues2Dim(Cov) #calcula autovalores
    EigVect = EigenVector2Dim(Cov, EigenValue) #calcula autovetores
    MatEigen = OrganizeEigen(EigenValue, EigVect) #organiza autovalores e vetores em ordem crescente de relevancia para analise
    FeatureVector = MatEigen[1:,:]
    FinalData = np.matmul(np.transpose(FeatureVector), np.transpose(M_data))
 
    M_data_pred_x = np.linspace(-10,10,20) * EigVect[0,0] #gera linha da PC1
    M_data_pred_y = M_data_pred_x * EigVect[1,0]
    M_data_pred_x2 = np.linspace(-7,7,20) * EigVect[0,1] #gera linha da PC2
    M_data_pred_y2 = M_data_pred_x2 * EigVect[1,1]

    
    
#    plt.figure()
    plt.scatter(M_data[:,0], M_data[:,1], color = 'blue')
    plt.plot(M_data_pred_x,M_data_pred_y, color = 'green')
    plt.plot(M_data_pred_x2,M_data_pred_y2, color = 'green')    
    plt.title('Dados de ponto de ebulição da água por P x Temperatura')
    plt.xlabel('Temperature (ºF)')
    plt.ylabel('BPt')
    plt.show()
    
#    plt.figure()
#    plt.scatter(M_data[:,0] + Mean.BPt, M_data[:,1] + Mean.Pressure, color = 'blue')
#    plt.plot(M_data_pred_x  + Mean.BPt, M_data_pred_y + Mean.Pressure, color = 'green')
#    plt.title('Dados de ponto de ebulição da água por P x Temperatura')
#    plt.xlabel('Temperature (ºF)')
#    plt.ylabel('BPt')
#    plt.show()
    
    

PCA(data)



