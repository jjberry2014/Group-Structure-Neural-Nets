#%%
import os, sys
homeDir = os.path.expanduser("~")
sys.path.insert(0,homeDir + "/codes/Group-Structure-Neural-Nets")
import Utilities as ut
import numpy as np
import xgboost as xgb
dataPath = homeDir + '/nuclear/data/GroupStructure/MOWA/run0002'
fileName = dataPath + '/LWR_data_7.mat'
#%%
kinfBOL,kinfMOL,kinfEOL,GS=ut.LoadData(fileName,1)
Nfeatures = 1000;
allData= ut.ProcessData(fileName, 1,1000,0,0,1)
# %%
kinf=np.array(np.zeros((len(kinfBOL),3)))
kinf[:,0]=kinfBOL
kinf[:,1]=kinfMOL
kinf[:,2]=kinfEOL#np.concatenate((kinfBOL,kinfMOL,kinfEOL),axis=0)
# %%
Nsamples,Ndecades = allData.shape
validationFrac = 0
testFrac = 0.2
X, X_test, y, y_test, vldF_corr = ut.makeFractions(Nsamples, validationFrac, testFrac, allData, kinf, 1)
# %%
dtrain = xgb.DMatrix(X, y)
dtest = xgb.DMatrix(X_test, y_test)
#param = {"max_depth": 20, "eta": 0.3, "objective": "reg:squarederror"}
#param = {"max_depth": 20, "eta": 0.7, "objective": "reg:squarederror", "booster":"gbtree"}
#param = {"max_depth": 7, "eta": 0.1, "objective": "reg:squarederror", "gamma":2, "colsample_bytree":0.5,"min_child_weight":7,"subsample":1}
param = {"max_depth": 20, "eta": 0.1, "objective": "reg:squarederror", "gamma":2, "colsample_bytree":0.5,"min_child_weight":0.5,"subsample":1}
watchlist = [(dtest, "eval"), (dtrain, "train")]
num_round = 1000
bst = xgb.train(param, dtrain, num_boost_round=num_round, evals=watchlist)
# %%
preds = bst.predict(dtest)
# %%
