import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, MaxPool2D, Dense, Flatten
from tensorflow.keras import Model

class MNET2():
    def get_model(self,input_shape=(32,32,1), numbers_layers_to_freeze=145):
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=input_shape,
            include_top=False,
            weights="imagenet")

        # Freeze the base_model
        base_model.trainable = True

        for i in range(numbers_layers_to_freeze):
            base_model.layers[i].trainable = False

        model = tf.keras.Sequential()
        model.add(base_model)
        model.add(tf.keras.layers.GlobalAveragePooling2D())
        model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
        model.summary()

        return model


class SimpleNet():
    def get_model(self,input_shape=(32,32,1), numbers_layers_to_freeze=145):
        start_filter = 16
        input = Input(shape=input_shape)
        layer = Conv2D(filters=start_filter, padding='same', kernel_size=(3, 3), activation='relu')(input)
        layer = MaxPool2D(pool_size=(2, 2))(layer)
        start_filter *= 2

        for i in range(3):
            layer = Conv2D(filters=start_filter, kernel_size=(3, 3), padding='same', activation='relu')(layer)
            layer = MaxPool2D(pool_size=(2, 2))(layer)
            start_filter *= 2

        layer = Flatten()(layer)
        layer = Dense(units=512, activation='relu')(layer)
        output = Dense(units=3, activation='softmax')(layer)

        model = Model(inputs=input, outputs=output)

        model.summary()
        return model


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Flatten, Dense, ReLU
from tensorflow.keras.layers import MaxPooling2D, MaxPool2D, AvgPool2D, Dropout
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.utils import plot_model

class ResNetStep():
    def get_model(self, input_shape=(32, 32, 1), numbers_layers_to_freeze=145):
        # Model building
        start_f = 32

        # Define model input
        input = Input(shape=input_shape)

        # Features extraction
        # First basic block : Conv layer (with 32 filters of 3x3) + MaxPool
        new_layer = Conv2D(start_f, (3, 3), padding='same', activation='relu')(input)
        new_layer = MaxPool2D(pool_size=(2, 2))(new_layer)

        # Second basic block : Conv layer (with 64 filters of 3x3) + MaxPool
        new_layer = Conv2D(64, (3, 3), padding='same', activation='relu')(new_layer)
        new_layer = MaxPool2D(pool_size=(2, 2))(new_layer)

        # Inception module
        new_layer = inception_module(new_layer, 64, 96, 128, 16, 32, 32)

        # Max Pool 2x2
        new_layer = MaxPool2D(pool_size=(2, 2))(new_layer)

        # Inception module
        new_layer = inception_module(new_layer, 128, 128, 256, 24, 64, 64)

        # Max Pool 2x2
        new_layer = MaxPool2D(pool_size=(2, 2))(new_layer)

        # Inception module
        new_layer = inception_module(new_layer, 384, 192, 384, 48, 128, 128)

        # Classifier
        new_layer = Flatten()(new_layer)
        new_layer = Dense(units=1024, activation='relu')(new_layer)
        output = Dense(units=3, activation='softmax')(new_layer)

        # Create the model
        model = Model(inputs=input, outputs=output)

        model.summary()
        return model


# An inception module is created by concatenating:
# 	A) 1 conv layer with 'f1' 1x1 filters
# 	B) 1 conv layer with 'f2_in' 1x1 filters followed by 1 conv layer with 'f2_out' 3x3 filters
# 	C) 1 conv layer with 'f3_in' 1x1 filters followed by 1 conv layer with 'f3_out' 5x5 filters
# 	D) 1 max pooling layer 3x3 followed by a 1 conv layer with 'f4_out' 1x1 filters

def inception_module(layer_in, f1, f2_in, f2_out, f3_in, f3_out, f4_out):
    # 1x1 conv
    conv1 = Conv2D(f1, (1, 1), padding='same', activation='relu')(layer_in)
    # 3x3 conv
    conv3 = Conv2D(f2_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv3 = Conv2D(f2_out, (3, 3), padding='same', activation='relu')(conv3)
    # 5x5 conv
    conv5 = Conv2D(f3_in, (1, 1), padding='same', activation='relu')(layer_in)
    conv5 = Conv2D(f3_out, (5, 5), padding='same', activation='relu')(conv5)
    # 3x3 max pooling
    pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(layer_in)
    pool = Conv2D(f4_out, (1, 1), padding='same', activation='relu')(pool)
    # concatenate filters, assumes filters/channels last
    layer_out = Concatenate(axis=-1)([conv1, conv3, conv5, pool])

    return layer_out

'''
Add more classes to create different model architectures. You will
use command line with the name of the class

class NewClass():
    def get_model(self, input_shape=..., ...):
        define model 
        return model
'''