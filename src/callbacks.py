import tensorflow as tf
def callbacks(patience=1000,min_delta=1e-8):
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=1e-8,
        patience=patience,
        verbose=2,
        mode="min",
        baseline=None,
        restore_best_weights=True)
    return callback