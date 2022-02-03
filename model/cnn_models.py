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
        input = Input(shape=(32, 32, 3))
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



'''
Add more classes to create different model architectures. You will
use command line with the name of the class

class NewClass():
    def get_model(self, input_shape=..., ...):
        define model 
        return model
'''