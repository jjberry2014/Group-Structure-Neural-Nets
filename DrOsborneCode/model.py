# %%
import tensorflow as tf
from tensorflow.keras import layers
# %%
def model(Nfeatures,dropout_rate=0.05):
    keff_model = tf.keras.Sequential([
        layers.Dense(256, activation='relu', name='hidden_1', input_dim=999),
        layers.Dropout(rate=dropout_rate),
        layers.Dense(256, activation='relu', name='hidden_2'),
        layers.Dropout(rate=dropout_rate),
        #layers.Dense(240, activation='relu',  name='hidden_3'),
        #layers.Dropout(rate=dropout_rate),
        layers.Dense(3, name='output')])
    return keff_model

class WeightRecordCallback(tf.keras.callbacks.Callback):
    def __init__(self, mlmodel):
        Nlayers = len(mlmodel.layers)
        for i in range(0,Nlayers):
            weightsi = mlmodel.layers[i].get_weights()
            print(weightsi)    
