# %%
import sys

sys.path.insert(0, '/Volumes/data/LosAlamosSummer')

import Utilities as ut
import importlib
import model as mod
import predict_with_uncertainty as pu
import custom as cus
importlib.reload(ut)
importlib.reload(mod)
importlib.reload(pu)
importlib.reload(cus)
import scipy
import tensorflow as tf
import numpy as np

# %%/Volumes/data/LosAlamosSummer/Utilities.py
datapath = '/Volumes/data/LosAlamosSummer/LWR/DATA/LWR_data_7.mat'
#datapath = '/Users/andrew/nuclear/data/GroupStructure/ANE-Paper-2021/NewDataSetFull1.mat'
print('Loading In Data')
fullSim=1
Nfeatures = 1000
kinfBOL,kinfMOL,kinfEOL,GS=ut.LoadData(datapath,1)
kinf=np.array(np.zeros((len(kinfBOL),3)))
kinf[:,0]=kinfBOL
kinf[:,1]=kinfMOL
kinf[:,2]=kinfEOL
#MakeGroupDensity(X, nDecades)
Nfeatures = 1000;
allData= ut.ProcessData(datapath, 1,Nfeatures,0,0,1)
# allData: (100,000x1,000) y_direct: (100,000x3)
# allData: (100,000x1,000) y_direct: (100,000x3)
print('Finished Loading Data')
#%%
dropout_rate = 0.00

# %% split the data
Nsamples,Ndecades = allData.shape
vldF=.1
testF=.2
normConst=1#np.linalg.norm(kinf)
y_norm=np.array(kinf/normConst)

X, x_test, y, y_test, vldF_corr = ut.makeFractions(Nsamples, vldF, testF, allData, y_norm, 1)


NtrainingSamples = int(Nsamples*(1 - testF))
tranValSplit=int(NtrainingSamples*(1-vldF_corr))
x_train=X[:tranValSplit,:]
y_train=y[:,:tranValSplit]
x_val=X[tranValSplit+1:,:]
y_val=y[:,tranValSplit+1:]# %% initialize callback

print(y_test.shape)
print(y_val.shape)
print(y_norm.shape)
print(y_train.shape)

model = mod.model(Nfeatures,dropout_rate)
model.build()
model.summary()
#cb = mod.WeightRecordCallback(model)
cbTb = tf.keras.callbacks.TensorBoard(histogram_freq=1)
#cbAmc = amc.Callback()
# %% do training
model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(1e-3))
model.fit(x_train,y_train.T, epochs=100, verbose=1,validation_data=(x_val,y_val.T), callbacks=cus.callbacks())
model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(1e-4))
model.fit(x_train,y_train.T, epochs=100, verbose=1,validation_data=(x_val,y_val.T), callbacks=cus.callbacks())
model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(1e-5))
model.fit(x_train,y_train.T, epochs=100, verbose=1,validation_data=(x_val,y_val.T), callbacks=cus.callbacks())
model.compile(loss='mean_absolute_error',optimizer=tf.keras.optimizers.Adam(1e-6))
model.fit(x_train,y_train.T, epochs=100, verbose=1,validation_data=(x_val,y_val.T), callbacks=cus.callbacks())
# %% predict with uncertainty
y_predicted, y_uncertainty, result = pu.predict_with_uncertainty(model,x_test)
# %% save results
scipy.io.savemat("out.mat",{"y_predicted":y_predicted,"y_uncertainty":y_uncertainty,"y_test":y_test})
# %%ompute MAE (0.0081 last)
metric = tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None)
metric.update_state(y_predicted,y_test.T)
metric.result().numpy()
# %% save model
model.save('./models/zero-dropout-0.0070-MAE')

# %%
