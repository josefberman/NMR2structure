from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.constraints import Constraint
from keras.layers import Input, Conv2D, Dense, Concatenate, MaxPool2D, Flatten, Reshape
from keras.models import Model
from keras.losses import Huber
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras_tuner.tuners import BayesianOptimization


def jaccard_index(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_pred = tf.cast(y_pred, tf.int32)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    return intersection / union


def hamming_distance(y_true, y_pred):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(tf.round(y_pred), tf.bool)
    return tf.reduce_mean(tf.cast(tf.not_equal(y_true, y_pred), tf.float32))


class ZeroConstraint(Constraint):
    def __init__(self, mask):
        self.mask = tf.convert_to_tensor(mask, dtype=tf.float32)

    def __call__(self, w):
        return w * self.mask


def encode_spectrum(input_array: np.array):
    # Encoder block
    input_train = input_array[:int(input_array.shape[0] * 0.9)]
    input_test = input_array[int(input_array.shape[0] * 0.9):]
    input_shape = input_train.shape[1:]
    encoder_inputs = Input(shape=input_shape, name='input_layer_encoder')
    x_0 = Flatten(name='flatten_layer_encoder')(encoder_inputs)
    x_1 = Dense(units=x_0.shape[1], activation=None, name='dense_layer_0_encoder')(x_0)
    x_2 = Dense(units=x_1.shape[1], activation=None, name='dense_layer_1_encoder')(x_1)
    x_3 = Dense(units=x_2.shape[1], activation=None, name='dense_layer_2_encoder')(x_2)
    encoder_outputs = Dense(units=512, activation=None, name='output_layer_encoder')(x_3)

    # Decoder block
    x_4 = Dense(units=x_3.shape[1], activation=None, name='dense_layer_1_decoder')(encoder_outputs)
    x_5 = Dense(units=x_2.shape[1], activation=None, name='dense_layer_2_decoder')(x_4)
    x_6 = Dense(units=x_1.shape[1], name='dense_layer_3_decoder')(x_5)
    x_6 = Dense(units=x_6.shape[1], name='dense_layer_4_decoder')(x_6)
    decoder_outputs = Reshape(input_shape, name='output_layer_decoder')(x_6)
    autoencoder = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    encoder = Model(inputs=encoder_inputs, outputs=encoder_outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss=Huber())
    print(autoencoder.summary())
    early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=5)
    autoencoder.fit(x=input_train, y=input_train, epochs=200, batch_size=32, validation_split=0.8,
                    callbacks=[early_stopping, reduce_lr])
    score = autoencoder.evaluate(input_test, input_test)
    print('Test loss: ', score)
    return encoder.predict(input_array)


def create_kernel_constraint(kernel_size: int, kernel_length: int, num_filters: int):
    mask = np.zeros((kernel_size, kernel_length, 1, num_filters))
    mask[0, :, :, :] = 1  # sets the first row in mask to 1, in each filter
    mask[-1, :, :, :] = 1  # sets the last row in mask to 1, in each filter
    return mask


def initialize_model(input_size: int, fingerprint_length: int = 167):
    input_layer = Input(shape=(input_size,))
    prior_output_layers = []
    for i in range(fingerprint_length):
        dense_layer = Dense(units=input_size, activation='tanh')(input_layer)
        while dense_layer.shape[1] > 1:
            dense_layer = Dense(units=dense_layer.shape[1] // 2, activation='tanh')(dense_layer)
        else:
            dense_layer = Dense(units=1, activation='tanh')(dense_layer)
        prior_output_layers.append(dense_layer)
    output_layer = Concatenate()(prior_output_layers)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-5), loss=Huber(), metrics=[jaccard_index, hamming_distance])
    print(model.summary())
    return model


def train_model(model: keras.Model, input_array: np.array, maccs_fingerprint: np.array):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=f'./logs/{timestamp}')
    early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(patience=3)
    model.fit(x=input_array, y=maccs_fingerprint, batch_size=32, epochs=1000,
              callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback], validation_split=0.15,
              shuffle=False)
    return model


def predict_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array):
    return model.predict(x=[carbon_input_array, proton_input_array])


def evaluate_model(model: keras.Model, input_array: np.array, maccs_fingerprint: np.array):
    return model.evaluate(x=input_array, y=maccs_fingerprint)

# def build_model(input_size, embedding_length, fingerprint_length, conv_filters_2, conv_filters, conv_kernel,
#                 conv_kernel_2, dense_units, dense_units_prior_output):
#     input_layer_carbon = Input(shape=(input_size, embedding_length, 1))
#     input_layer_proton = Input(shape=(input_size, embedding_length, 1))
#     conv_layer_carbon = Conv2D(filters=conv_filters, kernel_size=(conv_kernel, embedding_length), strides=(1, 1),
#                                padding='valid')(input_layer_carbon)
#     conv_layer_carbon_2 = Conv2D(filters=conv_filters_2, kernel_size=(conv_kernel_2, 1), strides=(1, 1),
#                                  padding='valid')(conv_layer_carbon)
#     maxpool_layer_carbon = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(conv_layer_carbon_2)
#     flatten_layer_carbon = Flatten()(maxpool_layer_carbon)
#     dense_layer_carbon = Dense(units=dense_units)(flatten_layer_carbon)
#     conv_layer_proton = Conv2D(filters=conv_filters, kernel_size=(conv_kernel, embedding_length), strides=(1, 1),
#                                padding='valid')(input_layer_proton)
#     conv_layer_proton_2 = Conv2D(filters=conv_filters_2, kernel_size=(conv_kernel_2, 1), strides=(1, 1),
#                                  padding='valid')(conv_layer_proton)
#     maxpool_layer_proton = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(conv_layer_proton_2)
#     flatten_layer_proton = Flatten()(maxpool_layer_proton)
#     dense_layer_proton = Dense(units=dense_units)(flatten_layer_proton)
#     concat_layer = Concatenate(axis=1)([dense_layer_carbon, dense_layer_proton])
#     dense_layer = Dense(units=dense_units_prior_output, activation='relu')(concat_layer)
#     output_layer = Dense(units=fingerprint_length, activation='sigmoid')(dense_layer)
#     model = keras.Model(inputs=[input_layer_carbon, input_layer_proton], outputs=output_layer)
#     model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=[jaccard_index])
#     return model
#
#
# def build_hp(hp):
#     conv_filters = hp.Int('conv_filters', min_value=20, max_value=100, step=10)
#     conv_filters_2 = hp.Int('conv_filters_2', min_value=20, max_value=100, step=10)
#     conv_kernel = hp.Int('conv_kernel', min_value=2, max_value=30, step=2)
#     conv_kernel_2 = hp.Int('conv_kernel_2', min_value=2, max_value=30, step=2)
#     dense_units = hp.Int('dense_units', min_value=10, max_value=100, step=10)
#     dense_units_prior_output = hp.Int('dense_units_prior_output', min_value=100, max_value=500, step=50)
#     model = build_model(200, 56, 167, conv_filters, conv_filters_2, conv_kernel, conv_kernel_2, dense_units,
#                         dense_units_prior_output)
#     return model
#
#
# def train_model_with_hp_tuning(carbon_input_array: np.array, proton_input_array: np.array, maccs_fingerprint: np.array):
#     carbon_input_train, carbon_input_validation, proton_input_train, proton_input_validation, maccs_fingerprint_train, \
#         maccs_fingerprint_validation = train_test_split(carbon_input_array, proton_input_array, maccs_fingerprint,
#                                                         test_size=0.15)
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#     tensorboard_callback = TensorBoard(log_dir=f'./hp_tuning/tensorboard/{timestamp}')
#     early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
#     reduce_lr_callback = ReduceLROnPlateau(patience=3)
#     tuner_nmr = BayesianOptimization(build_hp, seed=42, objective='val_loss', max_trials=200,
#                                      directory=f'./hp_tuning/{timestamp}/',
#                                      project_name='tuning_nmr')
#     print(tuner_nmr.search_space_summary())
#     tuner_nmr.search([carbon_input_train, proton_input_train], maccs_fingerprint_train, epochs=200, batch_size=32,
#                      validation_data=([carbon_input_validation, proton_input_validation], maccs_fingerprint_validation),
#                      callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback], shuffle=False)
#     best_nmr_hp = tuner_nmr.get_best_hyperparameters(1)[0]
#     print(f'Best hyper parameters:\n{best_nmr_hp.values}')
#     best_nmr_model = tuner_nmr.get_best_models(1)[0]
#     keras.models.save_model(best_nmr_model, f'./hp_tuning/saved_model/{timestamp}')
#     return best_nmr_model
