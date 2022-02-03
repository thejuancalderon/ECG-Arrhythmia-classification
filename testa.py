import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, ReLU, MaxPool2D, Dense, Flatten
from tensorflow.keras import Model

start_filter = 16
input = Input(shape=(32,32,3))
layer = Conv2D(filters=start_filter, padding='same',  kernel_size=(3,3),activation='relu')(input)
layer = MaxPool2D(pool_size=(2,2))(layer)
start_filter*=2

for i in range(3):
    layer = Conv2D(filters=start_filter, kernel_size=(3, 3),padding='same', activation='relu')(layer)
    layer = MaxPool2D(pool_size=(2, 2))(layer)
    start_filter *= 2

layer = Flatten()(layer)
layer = Dense(units=512, activation='relu')(layer)
output = Dense(units=3, activation='softmax')(layer)

model = Model(inputs=input, outputs=output)


model.summary()