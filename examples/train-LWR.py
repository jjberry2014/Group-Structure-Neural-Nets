#%%
import os, sys
homeDir = os.path.expanduser("~")
sys.path.insert(0,homeDir + "/codes/Group-Structure-Neural-Nets")
import importlib
import src.Utilities as ut
import numpy as np
import src.models as models
import src.loss as loss
importlib.reload(loss)
import src.callbacks as cb
importlib.reload(cb)
import tensorflow as tf
#%%
dataPath = homeDir + '/nuclear/data/GroupStructure/MOWA/run0002'
fileName = dataPath + '/LWR_data_7.mat'
#%%
kinfBOL,kinfMOL,kinfEOL,GS=ut.LoadData(fileName,1)
Nfeatures = 1000
allData= ut.ProcessData(fileName, 1,1000,0,0,1)
# %%
kinf=np.array(np.zeros((len(kinfBOL),3)))
kinf[:,0]=kinfBOL
kinf[:,1]=kinfMOL
kinf[:,2]=kinfEOL#np.concatenate((kinfBOL,kinfMOL,kinfEOL),axis=0)
# %%
#stepSize = np.logspace(-3,-7,num=50,base=10.0)
stepSize = np.logspace(-3,-7,num=5,base=10.0)
epochs = [10000,400,400,400,400]
batch_size = 1024
Nsamples,Ndecades = allData.shape
validationFrac = 0
testFrac = 0.2
X, X_test, y, y_test, vldF_corr = ut.makeFractions(Nsamples, validationFrac, testFrac, allData, kinf, 1)
y = y.transpose()
y_test = y_test.transpose()
#%% instantiate model
ANN_list = models.model_ensemble(input_dim=999,output_dim=3,num_models=1,activation='elu',layerConfiguration=[512,512],preFitScaler=None)
#%% train and save model
def doFit(model_list,stepSize,epochsi):
    for modeli in model_list:
        modeli.compile(optimizer=tf.keras.optimizers.Adam(stepSize),loss=loss.max_mean_reduce)
        modeli.fit(X,y,epochs=epochsi,batch_size=batch_size,verbose=1,shuffle=True,callbacks=cb.callbacks(),validation_data=(X_test,y_test))
for i in range(0,len(stepSize)):
    doFit(ANN_list,stepSize[i],epochs[i])
models.save_ensemble_model(ANN_list,'.')
#%% predict function
def predict(model_list,X):
    preds = tf.stack([modeli(X, training=True) for modeli in model_list], axis=0) #forward pass
    meany = tf.reduce_mean(preds,axis=0)
    return meany
#%%
y_pred = predict(ANN_list,X_test)
y_pred = y_pred.numpy()
# %%
#loss.max_mean_reduce(y_pred,y_test)
from matplotlib import pyplot as plt
a = (y_pred-y_test)
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(a[:,2],bins=500,alpha=0.75,label="EOL",color='r')
ax.hist(a[:,1],bins=500,alpha=0.25,label="MOL",color='b')
ax.hist(a[:,0],bins=500,alpha=0.25,label="BOL",color='g')
print(np.std(a[:,:],axis=0))
# %%
