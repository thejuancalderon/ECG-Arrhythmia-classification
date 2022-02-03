import tensorflow as tf


class MNET2():
    def get_model(self,input_shape=(32,32,3), numbers_layers_to_freeze=145):
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



'''
Add more classes to create different model architectures. You will
use command line with the name of the class

class NewClass():
    def get_model(self, input_shape=..., ...):
        define model 
        return model
'''