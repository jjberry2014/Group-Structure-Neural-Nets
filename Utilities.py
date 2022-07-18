#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 12:17:02 2020

@author: jessiejo
"""
from __future__ import unicode_literals, print_function, division
import scipy.io as sio
#from sklearn.preprocessing import StandardScaler
#import torch
import numpy as np
import random
import time
import math


import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
from sklearn.metrics import classification_report
#from simulated_annealing.optimize import SimulatedAnneal
#from SAGonzalo import SimulatedAnnealGonzalo



# auto-documenting header goes here

def LoadData(datapath, BU):
    if BU:
        matContents = sio.loadmat(datapath)
        
        kinfBOL = np.transpose(matContents['kinfBOL']).ravel()#Flattens array
        kinfMOL = np.transpose(matContents['kinfMOL']).ravel()#Flattens array
        kinfEOL = np.transpose(matContents['kinfEOL']).ravel()#Flattens array

        #CollXS_18_92235=matContents['collXSBOL_92235_18']  
        #d_CollXS_18_92238=matContents['d_CollXS_18_92238']
        #d_CollXS_1_92235=matContents['d_CollXS_1_92235']
        #d_CollXS_1_92238=matContents['d_CollXS_1_92238']
        #rng_state = np.random.get_state()
        #np.random.shuffle(kinf)
        groupStruct = np.transpose(matContents['groupStruct'])
        #np.random.set_state(rng_state)
        #np.random.shuffle(groupStruct)
        
        return kinfBOL,kinfMOL,kinfEOL, groupStruct,#d_CollXS_18_92238,d_CollXS_1_92235,d_CollXS_1_92238 #returns contents of file
    else:
        matContents = sio.loadmat(datapath)
        kinf = np.transpose(matContents['kinf']).ravel()#Flattens array
        groupStruct = np.transpose(matContents['groupStruct'])
        return kinf, groupStruct


def MakeGroupDensity(X, nDecades,mode,inputSerial):
    #decades=[-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    if mode == 1:
        decades=np.log10(inputSerial)
    else:
        decades = np.linspace(-3, 7, nDecades)
    #print(decades)
    
    x_lognorm = np.log10(X)
    decadeDensity=np.zeros((x_lognorm.shape[0], len(decades)-1))
    for N in range(0,x_lognorm.shape[0]):
        for i in range(0,len(decades)-2):
            for j in range(0,len(x_lognorm[N,:])):
                if x_lognorm[N,j] >= decades[i] and x_lognorm[N,j] < decades[i+1]:
                    decadeDensity[N,i]=decadeDensity[N,i]+1
    return decadeDensity
        

# auto-documenting header goes here
def ProcessData(datapath, percOfData, nDecades,mode,inputSerial,BU):
    if BU:
        kinfBOL,kinfMOL,kinfEOL, groupStruct = LoadData(datapath,BU)
    else:
        kinf, groupStruct = LoadData(datapath,BU)


    
    Nsamples,Ngroups = groupStruct.shape
    NtrainingSamples = int(round(Nsamples*percOfData))
    X = groupStruct[:NtrainingSamples,:(Ngroups)-1]
    #X_test = groupStruct[NtrainingSamples+1:,:(Ngroups)] # test data
    #scaler = StandardScaler() 
    #scaler.fit(X)
    #X = scaler.transform(X)
    #X_test = scaler.transform(X_test)
    
#goodKinfIndices = []#np.logical_and(kinf > 1.34134834087215,kinf < 1.42174834087215) #Takes good kinf indicies
    #good_18_235_ind= np.ravel(np.logical_and(CollXS_18_92235 > 49.3427304022287,CollXS_18_92235 < 52.3005498577954))
    # 500g value is 50.8216401300121
    #good_18_238_ind= np.ravel(np.logical_and(d_CollXS_18_92235 > 0.00,d_CollXS_18_92238 < 0.015171117038236))
    #good_1_235_ind= np.ravel(np.logical_and(d_CollXS_18_92235 > 0.00,d_CollXS_1_92235 < 13.9169799616074))
    #good_1_238_ind= np.ravel(np.logical_and(d_CollXS_18_92235 > 0.00,d_CollXS_1_92238 < 10.7403311948897))
    
    # 500-g value is 1.3815
    #goodKinfIndices = np.logical_and(kinf > 1.3765, kinf < 1.3865)  # Takes good kinf indicies
#NgoodKinfIndices = sum(goodKinfIndices==True)
#y = np.zeros((2,NtrainingSamples))
#y1 = np.zeros((2,NtrainingSamples))
    #y2 = np.zeros((2,NtrainingSamples))
    #y3 = np.zeros((2,NtrainingSamples))
    #y4 = np.zeros((2,NtrainingSamples))
    
#y[0,goodKinfIndices[:NtrainingSamples]] = 1
#y[1,np.logical_not(goodKinfIndices[:NtrainingSamples])] = 1
    
    #y1[0,good_18_235_ind[:NtrainingSamples]] = 1
    #y1[1,np.logical_not(good_18_235_ind[:NtrainingSamples])] = 1
    
    #y2[0,good_18_238_ind[:NtrainingSamples]] = 1
    #y2[1,np.logical_not(good_18_238_ind[:NtrainingSamples])] = 1
    
    #y3[0,good_1_235_ind[:NtrainingSamples]] = 1
    #y3[1,np.logical_not(good_1_235_ind[:NtrainingSamples])] = 1
    
    #y4[0,good_1_238_ind[:NtrainingSamples]] = 1
    #y4[1,np.logical_not(good_1_238_ind[:NtrainingSamples])] = 1

    
    trainingData=MakeGroupDensity(X,nDecades,mode,inputSerial)
            
    return trainingData#, y, goodKinfIndices #,y2,y3,y4,good_18_235_ind,good_18_238_ind,good_1_235_ind,good_1_238_ind


# auto-documenting header goes here
#def randomChoice(l):
#    return random.randint(0, len(l) - 1)

# auto-documenting header goes here
#def randomTrainingExample(trainingData,y,goodKinfIndices):
#    all_categories = [True, False] 
#    randIntChoice = randomChoice(trainingData[:,0]) #Random Integer
#    category=goodKinfIndices[randIntChoice] #"Goodness" of Group Structure
#    GS = trainingData[randIntChoice,:] #One Random Group Structure
#    GS.shape=(len(GS),1)
#    category_tensor = torch.zeros(1, dtype=torch.long) # [1x]
#    category_tensor[0] = all_categories.index(category)# [1x]                                   
#    GS_tensor = torch.from_numpy(GS).to(torch.float)   # [11x1]
#    GS_tensor = GS_tensor.unsqueeze(1)                 # [11x1x1]
#    return category, GS, category_tensor, GS_tensor

# auto-documenting header goes here
#def categoryFromOutput(output):
#    all_categories = [True, False] 
#    top_n, top_i = output.topk(1)
#    category_i = top_i[0].item()
#    return all_categories[category_i], category_i

# auto-documenting header goes here
#def timeSince(since):
#    now = time.time()
#    s = now - since
 #   m = math.floor(s / 60)
#    s -= m * 60
#    return '%dm %ds' % (m, s)

# auto-documenting header goes here
def makeFractions(Nsamples, vldF, testF, trainingData, trainingAnswers,BU):
    if BU:
        NtrainingSamples = int(Nsamples*(1 - testF))

        vldF_corr=vldF*Nsamples/NtrainingSamples

        X = trainingData[:NtrainingSamples,:]
        y = trainingAnswers[:NtrainingSamples,:].T
        X_test = trainingData[NtrainingSamples+1:,:] # test data
    
        y_test = trainingAnswers[NtrainingSamples+1:,:].T
    else:
        NtrainingSamples = int(Nsamples*(1 - testF))
        
        vldF_corr=vldF*Nsamples/NtrainingSamples
        
        X = trainingData[:NtrainingSamples,:]
        y = trainingAnswers[:NtrainingSamples].T
        X_test = trainingData[NtrainingSamples+1:,:] # test data
        
        y_test = trainingAnswers[NtrainingSamples+1:].T

    return X, X_test, y, y_test ,vldF_corr

# auto-documenting header goes here
def numRight(yPred, y_test):
    countRight=0
    countWrong=0
    indexRight=[]
    indexWrong=[]
    for i in range(0,len(yPred[:,0])):
        if (yPred[i,:]==y_test[i,:]).all():
            indexRight.append(i)
            countRight=countRight+1
            
            
        else:
            indexWrong.append(i)
            countWrong=countWrong+1
            
    return countRight, countWrong, indexRight, indexWrong


def saveAsAMat(countRight, countWrong, lossCurve,yPredProba, solvertype,HL_sz,activationtype, percOfData,l,indexRight,indexWrong,y):    
    obj_arr = np.zeros((7,), dtype=np.object)
    obj_arr[0]=countRight
    obj_arr[1]=countWrong
    obj_arr[2]=lossCurve
    obj_arr[3]=yPredProba
    obj_arr[4]=indexRight
    obj_arr[5]=indexWrong
    obj_arr[6]=y
    
    sio.savemat(('/Volumes/data/VBUDS/GroupStructurePaper/NeuralNetworks/Data4Paper/Variables_%s_HL%d_at_%s_%gPerc_%g.mat'%(solvertype ,HL_sz,activationtype,percOfData,l)), mdict={'allVars': obj_arr})
    
    
def ErrorHistogram(yPred, y_test):
    
    Error=abs((yPred-y_test))/yPred
    
    plt.hist(Error, bins = 10)
    plt.show()

    return 



def OptimizeHyperparameters(X, X_test, y, y_test,gpr):
    
 
# This is the hyperparameter space we'll be searching over
    hl_sizes = [(4,4),(8,8),(16,16),(4),(8),(16)]
    
    neural_net_hyperparams = {'hidden_layer_sizes':hl_sizes,
                              #'Learning_Rate':np.arange(0.001, 1.001, 0.001),
                              'alpha':np.arange(0.1, 4.01, 0.1)
                              }


# Using a linear MLPRegressor             
# Initialize Simulated Annealing and fit
    sa = SimulatedAnnealGonzalo(gpr, neural_net_hyperparams, T=10.0, T_min=0.001, alpha=0.75,
                         verbose=True, max_iter=1, n_trans=5, max_runtime=300,
                         cv=3, scoring='max_error', refit=True)
    sa.fit(X, y)
# Print the best score and the best params
    print(sa.best_score_, sa.best_params_)
# Use the best estimator to predict classes
    optimized_gpr = sa.best_estimator_
    y_test_pred = optimized_gpr.predict(X_test)
# Print a report of precision, recall, f1_score
    #print(classification_report(y_test, y_test_pred))
    
    
    return sa.best_score_, sa.best_params_

    
def GroupDensityChecker(decadeDensity, push,origGS):
    newGS=[]
    count=0
    for i in range(0,len(decadeDensity)):
        if decadeDensity[i]>1: #Catch for if theres more than one count in the serailization structure
            for j in range(0,int(decadeDensity[i])):
                if push==1:
                    newGS.append(origGS[count+1])
                elif push==2:
                    newGS.append(origGS[count])
                elif push==3:
                    newGS.append(np.mean([origGS[count+1],origGS[count]]))
                else:
                    print("Push value not recognized")
            count=count+1
        elif decadeDensity[i]==1: #Only one count 
            if push==1:
                newGS.append(origGS[count+1])
            elif push==2:
                newGS.append(origGS[count])
            elif push==3:
                newGS.append(np.mean([origGS[count+1],origGS[count]]))
            else:
                print("Push value not recognized")
            count=count+1
        else:
            print("No count")
    return newGS







