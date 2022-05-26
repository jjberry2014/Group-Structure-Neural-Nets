import numpy as np

def predict_with_uncertainty(model,X,Niters=50):
    result = []
    for i in range(Niters):
        resulti = model(X,training=True).numpy()
        result.append(resulti)
    result = np.array(result)
    prediction = result.mean(axis=0)
    uncertainty = result.std(axis=0) 
    return prediction, uncertainty, result

