from datetime import datetime

import keras
import numpy as np
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.constraints import Constraint
from keras.layers import Input, Conv2D, Dense, Concatenate, MaxPool2D, Flatten
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras_tuner.tuners import BayesianOptimization


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
    conv_layer_carbon = Conv2D(filters=90, kernel_size=(10, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_carbon)
    maxpool_layer_carbon = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(conv_layer_carbon)
    flatten_layer_carbon = Flatten()(maxpool_layer_carbon)
    dense_layer_carbon = Dense(units=100)(flatten_layer_carbon)
    conv_layer_proton = Conv2D(filters=90, kernel_size=(10, embedding_length), strides=(1, 1), padding='valid')(
        input_layer_proton)
    maxpool_layer_proton = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(conv_layer_proton)
    flatten_layer_proton = Flatten()(maxpool_layer_proton)
    dense_layer_proton = Dense(units=100)(flatten_layer_proton)
    concat_layer = Concatenate(axis=1)([dense_layer_carbon, dense_layer_proton])
    dense_layer = Dense(units=400, activation='relu')(concat_layer)
    output_layer = Dense(units=fingerprint_length, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=[input_layer_carbon, input_layer_proton], outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=[jaccard_index])
    print(model.summary())
    return model


def train_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array,
                maccs_fingerprint: np.array):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=f'./logs/{timestamp}')
    early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(patience=3)
    model.fit(x=[carbon_input_array, proton_input_array], y=maccs_fingerprint, batch_size=32, epochs=100,
              callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback], validation_split=0.15)
    return model


def predict_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array):
    return model.predict(x=[carbon_input_array, proton_input_array])


def evaluate_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array,
                   maccs_fingerprint: np.array):
    return model.evaluate(x=[carbon_input_array, proton_input_array], y=maccs_fingerprint)


def build_model(input_size, embedding_length, fingerprint_length, conv_filters_2, conv_filters, conv_kernel,
                conv_kernel_2, dense_units, dense_units_prior_output):
    input_layer_carbon = Input(shape=(input_size, embedding_length, 1))
    input_layer_proton = Input(shape=(input_size, embedding_length, 1))
    conv_layer_carbon = Conv2D(filters=conv_filters, kernel_size=(conv_kernel, embedding_length), strides=(1, 1),
                               padding='valid')(input_layer_carbon)
    conv_layer_carbon_2 = Conv2D(filters=conv_filters_2, kernel_size=(conv_kernel_2, 1), strides=(1, 1),
                                 padding='valid')(conv_layer_carbon)
    maxpool_layer_carbon = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(conv_layer_carbon_2)
    flatten_layer_carbon = Flatten()(maxpool_layer_carbon)
    dense_layer_carbon = Dense(units=dense_units)(flatten_layer_carbon)
    conv_layer_proton = Conv2D(filters=conv_filters, kernel_size=(conv_kernel, embedding_length), strides=(1, 1),
                               padding='valid')(input_layer_proton)
    conv_layer_proton_2 = Conv2D(filters=conv_filters_2, kernel_size=(conv_kernel_2, 1), strides=(1, 1),
                                 padding='valid')(conv_layer_proton)
    maxpool_layer_proton = MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid')(conv_layer_proton_2)
    flatten_layer_proton = Flatten()(maxpool_layer_proton)
    dense_layer_proton = Dense(units=dense_units)(flatten_layer_proton)
    concat_layer = Concatenate(axis=1)([dense_layer_carbon, dense_layer_proton])
    dense_layer = Dense(units=dense_units_prior_output, activation='relu')(concat_layer)
    output_layer = Dense(units=fingerprint_length, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=[input_layer_carbon, input_layer_proton], outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=[jaccard_index])
    return model


def build_hp(hp):
    conv_filters = hp.Int('conv_filters', min_value=20, max_value=100, step=10)
    conv_filters_2 = hp.Int('conv_filters_2', min_value=20, max_value=100, step=10)
    conv_kernel = hp.Int('conv_kernel', min_value=2, max_value=30, step=2)
    conv_kernel_2 = hp.Int('conv_kernel_2', min_value=2, max_value=30, step=2)
    dense_units = hp.Int('dense_units', min_value=10, max_value=100, step=10)
    dense_units_prior_output = hp.Int('dense_units_prior_output', min_value=100, max_value=500, step=50)
    model = build_model(200, 56, 167, conv_filters, conv_filters_2, conv_kernel, conv_kernel_2, dense_units, dense_units_prior_output)
    return model


def train_model_with_hp_tuning(carbon_input_array: np.array, proton_input_array: np.array, maccs_fingerprint: np.array):
    carbon_input_train, carbon_input_validation, proton_input_train, proton_input_validation, maccs_fingerprint_train, \
        maccs_fingerprint_validation = train_test_split(carbon_input_array, proton_input_array, maccs_fingerprint,
                                                        test_size=0.15)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=f'./hp_tuning/tensorboard/{timestamp}')
    early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(patience=3)
    tuner_nmr = BayesianOptimization(build_hp, seed=42, objective='val_loss', max_trials=200,
                                     directory=f'./hp_tuning/{timestamp}/',
                                     project_name='tuning_nmr')
    print(tuner_nmr.search_space_summary())
    tuner_nmr.search([carbon_input_train, proton_input_train], maccs_fingerprint_train, epochs=200, batch_size=32,
                     validation_data=([carbon_input_validation, proton_input_validation], maccs_fingerprint_validation),
                     callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback])
    best_nmr_hp = tuner_nmr.get_best_hyperparameters(1)[0]
    print(f'Best hyper parameters:\n{best_nmr_hp.values}')
    best_nmr_model = tuner_nmr.get_best_models(1)
    keras.models.save_model(best_nmr_model, f'./hp_tuning/saved_model/{timestamp}')
    return best_nmr_model
