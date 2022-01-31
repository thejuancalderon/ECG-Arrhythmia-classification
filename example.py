import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField, MarkovTransitionField, RecurrencePlot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import tensorflow as tf
from data.image_generator import get_generators
from model.mobileNet import get_model

LABEL = "label"
NORMAL = "N"
VENTRICULAR = "V"
SUPER_VENTRICULAR = "S"

'''
Small example script to train a model
'''

if __name__ == '__main__':
    import gdown
    import os

    id = '1r8gX-S3u39IJyPV0y1-pUiykOnJeijug'
    output = 'dataset.csv'

    #% Check if dataset is in the folder otherwise downloads it
    if not os.path.isfile("dataset.csv"):
        gdown.download(id=id, output=output, quiet=False)
    # Get the generators
    training_generator, validation_generator = get_generators("dataset.csv")
    model = get_model()

    # Compile model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4), metrics='accuracy')
    # Callbacks
    checkpoint_filepath = 'drive/MyDrive/tensorboard.{epoch:02d}-{val_loss:.2f}'
    ck_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    tb_filepath = "drive/MyDrive/tensorboard"
    tb_callback = tf.keras.callbacks.TensorBoard(tb_filepath, update_freq=1)

    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=10)  # The first run was with val loss

    # Train the model
    model.fit_generator(generator=training_generator, validation_data=validation_generator, epbochs=100,
                        callbacks=[es_callback, tb_callback, ck_callback])

