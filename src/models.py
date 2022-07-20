import tensorflow as tf
from tensorflow.keras import layers

# %% ensemble model
def model_ensemble(input_dim=57,output_dim=57, num_models=5, activation='relu', layerConfiguration=[512,512], preFitScaler=None):
    #activation = 'softplus' # not bad!
    #activation = 'relu' # standard
    #activation = 'elu' # not good for UE
    NhiddenLayers = len(layerConfiguration)
    def create_single_model():
        Th = []
        if preFitScaler is not None:
            scaleMean = preFitScaler.mean_
            scaleVar = preFitScaler.var_
            Th.append(layers.Normalization(axis=1, mean=scaleMean,variance=scaleVar,input_dim=input_dim))
        Th.append(layers.Dense(layerConfiguration[0], activation=activation, name='hidden_1', input_dim=input_dim))
        for i in range(1,NhiddenLayers):
            Th.append(layers.Dense(layerConfiguration[i], activation=activation,  name='hidden_'+ str(i+1)))
        #Th.append(edl.layers.DenseNormal(output_dim))
        Th.append(layers.Dense(output_dim, name='output'))
        ss_model = tf.keras.Sequential(Th)
        ss_model.build()
        return ss_model
    models = [create_single_model() for _ in range(num_models)]
    models[0].summary()
    return models    
# %% save ensemble model
def save_ensemble_model(model_list, rootDir):
    counter = 0
    for model in model_list:
        model.save_weights(rootDir + '/model' + str(counter) + '/')
        counter += 1
# %% load a saved model
def load_ensemble_model(rootDir, numModels, activation='relu', layerConfiguration=[512,512],input_dim=999,output_dim=3):
    counter = 0
    model_list = model_ensemble(input_dim=input_dim,output_dim=output_dim, num_models=numModels, activation=activation, layerConfiguration=layerConfiguration)
    for model in model_list:
        model.load_weights(rootDir + '/model' + str(counter) + '/')
        counter += 1
    return model_list 

