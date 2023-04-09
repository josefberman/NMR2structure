import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, Conv2D, Dense, Concatenate, MaxPool2D, Flatten
from keras.callbacks import TensorBoard
from keras.constraints import Constraint
from datetime import datetime


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


def initialize_model(input_size: int, embedding_length: int, fingerprint_length: int = 166):
    input_layer_carbon = Input(shape=(input_size, embedding_length, 1))
    input_layer_proton = Input(shape=(input_size, embedding_length, 1))
    conv_layer_1_carbon = Conv2D(filters=10, kernel_size=(1, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_carbon)
    conv_layer_2_carbon = Conv2D(filters=10, kernel_size=(2, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_carbon)
    conv_layer_3_carbon = Conv2D(filters=10, kernel_size=(3, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_carbon)
    conv_layer_4_carbon = Conv2D(filters=10, kernel_size=(4, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_carbon)
    conv_layer_1_proton = Conv2D(filters=10, kernel_size=(1, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_proton)
    conv_layer_2_proton = Conv2D(filters=10, kernel_size=(2, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_proton)
    conv_layer_3_proton = Conv2D(filters=10, kernel_size=(3, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_proton)
    conv_layer_4_proton = Conv2D(filters=10, kernel_size=(4, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_proton)
    """
    concat_layer_carbon = Concatenate(axis=1)(
        [conv_layer_1_carbon, conv_layer_2_carbon, conv_layer_3_carbon, conv_layer_4_carbon])
    maxpool_layer_carbon = MaxPool2D(pool_size=(4, 1), strides=(2, 1), padding='valid')(concat_layer_carbon)
    concat_layer_proton = Concatenate(axis=1)(
        [conv_layer_1_proton, conv_layer_2_proton, conv_layer_3_proton, conv_layer_4_proton])
    maxpool_layer_proton = MaxPool2D(pool_size=(4, 1), strides=(2, 1), padding='valid')(concat_layer_proton)
    conv_layer_1_carbon = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_carbon)
    conv_layer_2_carbon = Conv2D(filters=10, kernel_size=(2, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_carbon)
    conv_layer_3_carbon = Conv2D(filters=10, kernel_size=(3, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_carbon)
    conv_layer_4_carbon = Conv2D(filters=10, kernel_size=(4, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_carbon)
    conv_layer_1_proton = Conv2D(filters=10, kernel_size=(1, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_proton)
    conv_layer_2_proton = Conv2D(filters=10, kernel_size=(2, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_proton)
    conv_layer_3_proton = Conv2D(filters=10, kernel_size=(3, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_proton)
    conv_layer_4_proton = Conv2D(filters=10, kernel_size=(4, 1), strides=(1, 1), padding='valid')(
        maxpool_layer_proton)
    """
    concat_layer_all = Concatenate(axis=1)(
        [conv_layer_1_carbon, conv_layer_2_carbon, conv_layer_3_carbon, conv_layer_4_carbon, conv_layer_1_proton,
         conv_layer_2_proton, conv_layer_3_proton, conv_layer_4_proton])
    maxpool_layer_all = MaxPool2D(pool_size=(4, 1), strides=(1, 1), padding='valid')(concat_layer_all)
    flatten_layer = Flatten()(maxpool_layer_all)
    dense_layer = Dense(units=fingerprint_length * 2, activation='relu')(flatten_layer)
    dense_layer = Dense(units=fingerprint_length * 2, activation='relu')(dense_layer)
    output_layer = Dense(units=fingerprint_length, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=[input_layer_carbon, input_layer_proton],
                        outputs=output_layer)
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
