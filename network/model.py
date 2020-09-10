import tensorflow as tf


def model_test_1(time_step):
    input_layer = tf.keras.layers.Input([time_step, 1], dtype=tf.float32)

    gru = tf.keras.layers.LSTM(256)(input_layer)

    # use 2 dense layer
    output = tf.keras.layers.Dense(512)(gru)
    output = tf.keras.layers.Dense(256)(output)
    output = tf.keras.layers.Dropout(rate=0.1)(output)
    output = tf.keras.layers.Dense(128)(output)
    output = tf.keras.layers.Dropout(rate=0.1)(output)
    output = tf.keras.layers.Dense(2)(output)

    return tf.keras.Model(inputs=input_layer, outputs=output)


def model_test_1_2(time_step):
    input_layer = tf.keras.layers.Input([time_step, 1])

    output = tf.keras.layers.Bidirectional(layer=tf.keras.layers.GRU(10, return_sequences=True))(input_layer)
    output = tf.keras.layers.Dense(1)(output)
    output = tf.keras.layers.Flatten()(output)
    # output = tf.keras.layers.Dense(time_step)(output)
    # output = tf.keras.layers.Dropout(0.3)(output)
    output = tf.keras.layers.Dense(time_step, activation=tf.keras.activations.softmax)(output)

    return tf.keras.Model(inputs=input_layer, outputs=output)


def model_test_1_1(time_step):
    input_layer = tf.keras.layers.Input([time_step, 1])

    gru = tf.keras.layers.LSTM(2)(input_layer)

    return tf.keras.Model(inputs=input_layer, outputs=gru)


def model_test_2(width, height):
    input_layer = tf.keras.layers.Input([width, height, 1])

    output = tf.keras.layers.Conv2D(32, [1, 1])(input_layer)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.Conv2D(64, [1, 1])(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.Conv2D(128, [1, 1])(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.Conv2D(256, [1, 1])(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.MaxPool2D([1, 2])(output)
    output = tf.keras.layers.Conv2D(256, [1, 1])(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.MaxPool2D()(output)
    output = tf.keras.layers.Dense(256)(output)
    output = tf.keras.layers.Dense(256)(output)
    output = tf.keras.layers.Dense(2)(output)  # x, y

    return tf.keras.Model(inputs=input_layer, outputs=output)


def model_test_3():
    input_layer = tf.keras.layers.Input((227, 227, 1))
    # 1st conv
    output = tf.keras.layers.Conv2D(96, (11, 11), strides=(4, 4), activation='relu', input_shape=(227, 227, 3))(input_layer)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))(output)
    # 2nd conv
    output = tf.keras.layers.Conv2D(256, (11, 11), strides=(1, 1), activation='relu', padding="same")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    # 3rd conv
    output = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    # 4th conv
    output = tf.keras.layers.Conv2D(384, (3, 3), strides=(1, 1), activation='relu', padding="same")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    # 5th Conv
    output = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding="same")(output)
    output = tf.keras.layers.BatchNormalization()(output)
    output = tf.keras.layers.MaxPooling2D(2, strides=(2, 2))(output)
    # To Flatten layer
    output = tf.keras.layers.Flatten()(output)

    return tf.keras.Model(inputs=input_layer, outputs=output)
