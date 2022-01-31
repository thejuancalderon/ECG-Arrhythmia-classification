import tensorflow as tf
def get_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32,32,3),
        include_top=False,
        weights="imagenet")

    # Freeze the base_model
    base_model.trainable = True

    for i in range(145):
      base_model.layers[i].trainable = False

    model = tf.keras.Sequential()
    model.add(base_model)
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(tf.keras.layers.Dense(units=3, activation='softmax'))
    model.summary()

    return model