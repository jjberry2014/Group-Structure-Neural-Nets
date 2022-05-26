import tensorflow as tf

def learning_schedule(decay_steps=1000,decay_rate=0.96):
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps,
        decay_rate,
        staircase=True)
    return lr_schedule

def callbacks():
    callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.00001,
        patience=15,
        verbose=2,
        mode="auto",
        baseline=None,
        restore_best_weights=True)
    return callback