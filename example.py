import tensorflow as tf
from data.image_generator import get_generators
import argparse
import gdown
import os
import datetime
import importlib

LABEL = "label"
NORMAL = "N"
VENTRICULAR = "V"
SUPER_VENTRICULAR = "S"
# Url dataset
ID = '1r8gX-S3u39IJyPV0y1-pUiykOnJeijug'
OUTPUT = 'dataset.csv'

'''
Small example script to train a model
'''

if __name__ == '__main__':

    # Parsing arguments from commandline. Useful if we want to run multiple experiments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, required=False, nargs='?', default=32, const=32)
    parser.add_argument('--bs', type=int, required=False, nargs='?', default=512, const=512)
    parser.add_argument('--lr', type=float, required=False, nargs='?', default=3e-4, const=3e-4)
    parser.add_argument('--layers_to_freeze', type=int, required=False, nargs='?', default=145, const=145)
    parser.add_argument('--model', type=str, required=False, nargs='?', default='MNET2', const='MNET2', help='model name. Use the name of the class')
    parser.add_argument('--os_card', type=int, required=False, nargs='?', default=100000, const=100000, help='cardinality over sampling')
    parser.add_argument('--us_card', type=int, required=False, nargs='?', default=100000, const=100000, help='cardinality under sampling')
    args = parser.parse_args()


    #% Check if dataset is in the folder otherwise downloads it
    if not os.path.isfile("dataset.csv"):
        gdown.download(id=ID, output=OUTPUT, quiet=False)

    # Get the generators
    training_generator, validation_generator = get_generators("dataset.csv",
                                                              oversampling_cardinality=args.os_card,
                                                              undersampling_cardinality=args.us_card,
                                                              input_size=(args.img_size,args.img_size,3))

    # Get the model. We use the name from cmd to retrieve the correct class (using importlib to dynamically linked the class)
    #Import the module
    module = importlib.import_module("model")
    # Import the class
    model_class = getattr(module, args.model)
    # Use get_model function of the class to retrieve the model or the keras neural network
    model = model_class.get_model(model_class, input_shape=(args.img_size,args.img_size,3), numbers_layers_to_freeze=args.layers_to_freeze)

    # Compile model
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr), metrics='accuracy')
    # Callbacks ckpt
    ckpt_path = str(datetime.datetime.now())
    os.makedirs('experiments/' + ckpt_path)
    checkpoint_filepath = 'experiments/' + ckpt_path + '/check.{epoch:02d}-{val_loss:.2f}'
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
    model.fit(x=training_generator, validation_data=validation_generator, epochs=100,
                        callbacks=[es_callback, ck_callback])

