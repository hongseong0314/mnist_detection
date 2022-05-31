import tensorflow as tf


def load_model(img_size=(28,28)):
    """
    model 
    """
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=img_size),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
        #tf.keras.layers.Dense(10, activation='relu')
        ])
    return model