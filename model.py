import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Concatenate, MaxPool2D, Flatten
from keras.callbacks import TensorBoard
from keras.constraints import Constraint
from datetime import datetime


def jaccard_index(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_pred = tf.cast(y_pred, tf.int32)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    return intersection / union


class ZeroConstraint(Constraint):
    def __init__(self, mask):
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def __call__(self, w):
        return w * self.mask


def create_kernel_constraint(kernel_size: int, kernel_length: int, num_filters: int):
    mask = np.zeros((kernel_size, kernel_length, 1, num_filters))
    mask[0, :, :, :] = 1  # sets the first row in mask to 1, in each filter
    mask[-1, :, :, :] = 1  # sets the last row in mask to 1, in each filter
    return mask


def initialize_model(input_size: int, embedding_length: int, fingerprint_length: int = 167):
    input_layer_carbon = Input(shape=(input_size, embedding_length, 1))
    input_layer_proton = Input(shape=(input_size, embedding_length, 1))
    conv_layer_carbon = Conv2D(filters=10, kernel_size=(16, embedding_length), strides=(1, 1), padding='same')(
        input_layer_carbon)
    maxpool_layer_carbon = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_layer_carbon)
    flatten_layer_carbon = Flatten()(maxpool_layer_carbon)
    dense_layer_carbon = Dense(units=30)(flatten_layer_carbon)
    conv_layer_proton = Conv2D(filters=10, kernel_size=(16, embedding_length), strides=(1, 1), padding='same')(
        input_layer_proton)
    maxpool_layer_proton = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(conv_layer_proton)
    flatten_layer_proton = Flatten()(maxpool_layer_proton)
    dense_layer_proton = Dense(units=30)(flatten_layer_proton)
    concat_layer = Concatenate(axis=1)([dense_layer_carbon, dense_layer_proton])
    dense_layer = Dense(units=fingerprint_length, activation='relu')(concat_layer)
    output_layer = Dense(units=fingerprint_length, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=[input_layer_carbon, input_layer_proton], outputs=output_layer)
    model.compile(optimizer='adam', loss='mse', metrics='accuracy')
    print(model.summary())
    return model


def train_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array,
                maccs_fingerprint: np.array):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=f'./logs/{timestamp}', write_graph=True, write_images=True)
    model.fit(x=[carbon_input_array, proton_input_array], y=maccs_fingerprint, batch_size=32, epochs=100,
              callbacks=[tensorboard_callback], validation_split=0.15)
    return model


def predict_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array):
    return model.predict(x=[carbon_input_array, proton_input_array])


def evaluate_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array,
                   maccs_fingerprint: np.array):
    return model.evaluate(x=[carbon_input_array, proton_input_array], y=maccs_fingerprint)
