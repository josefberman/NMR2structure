from datetime import datetime

import keras
import numpy as np
import rdkit
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.constraints import Constraint
from keras.layers import Input, Conv2D, Dense, Concatenate, MaxPool2D, Flatten, Reshape, BatchNormalization, Conv1D, \
    ReLU, Add, GlobalAveragePooling1D, Dropout
from keras.models import Model
from keras.losses import Huber
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras_tuner.tuners import BayesianOptimization
from keras import backend as K
from sklearn.model_selection import KFold
import os
from sklearn.preprocessing import normalize


def encoder_cosine_similarity(y_true, y_pred):
    y_true = y_true / tf.sqrt(tf.cast(tf.reduce_sum(tf.pow(y_true, 2)), float))
    y_pred = y_pred / tf.sqrt(tf.cast(tf.reduce_sum(tf.pow(y_pred, 2)), float))
    return tf.reduce_sum(y_true * y_pred)


def jaccard_index(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return intersection.sum() / float(union.sum())
    # y_pred = tf.cast(y_pred > 0.5, tf.float32)
    # y_pred = tf.cast(y_pred, tf.int32)
    # intersection = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    # union = tf.reduce_sum(tf.cast(tf.logical_or(tf.equal(y_true, 1), tf.equal(y_pred, 1)), tf.float32))
    # return intersection / union


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
    encoder_inputs = Input(shape=(*input_shape, 1))
    x_0 = Flatten(name='flatten_layer_encoder')(encoder_inputs)
    x_1 = Dense(units=512)(x_0)
    x_2 = Dense(units=256)(x_1)
    x_3 = Dense(units=128)(x_2)
    x_4 = Dense(units=64)(x_3)
    encoder_outputs = Dense(units=32, name='output_layer_encoder')(x_4)

    # Decoder block
    x_4 = Dense(units=64)(encoder_outputs)
    x_5 = Dense(units=128)(x_4)
    x_6 = Dense(units=256)(x_5)
    x_7 = Dense(units=512)(x_6)
    x_8 = Dense(units=x_0.shape[1])(x_7)
    decoder_outputs = Reshape(input_shape, name='output_layer_decoder')(x_8)
    autoencoder = Model(inputs=encoder_inputs, outputs=decoder_outputs)
    encoder = Model(inputs=encoder_inputs, outputs=encoder_outputs)
    autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mse', metrics=[encoder_cosine_similarity])
    print(autoencoder.summary())
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(patience=3, min_delta=1e-6)
    encoder_tensorboard = TensorBoard(log_dir='./encoder_logs/')
    if not os.path.exists('./saved_encoder_model'):
        autoencoder.fit(x=input_train, y=input_train, epochs=1000, batch_size=32, validation_split=0.8, shuffle=True,
                        callbacks=[early_stopping, encoder_tensorboard, reduce_lr])
        autoencoder.save('./saved_encoder_model/')
    else:
        autoencoder = keras.models.load_model('./saved_encoder_model/',
                                              custom_objects={'encoder_cosine_similarity': encoder_cosine_similarity})
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
    dense_layer = Dense(units=input_size, activation='relu')(input_layer)
    dense_layer = Dense(units=input_size, activation='relu')(dense_layer)
    dense_layer = Dense(units=input_size, activation='relu')(dense_layer)
    output_layer = Dense(units=fingerprint_length, activation='sigmoid')(dense_layer)
    model = keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=Adam(learning_rate=1e-6), loss=Huber(), metrics=[hamming_distance])
    # print(model.summary())
    return model


def train_model(model: keras.Model, input_array: np.array, maccs_fingerprint: np.array):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=f'./logs/{timestamp}')
    early_stopping_callback = EarlyStopping(patience=5, restore_best_weights=True)
    reduce_lr_callback = ReduceLROnPlateau(patience=3)
    metric_per_fold = []
    loss_per_fold = []
    kfold = KFold(n_splits=5, shuffle=True)
    for i, (k_train, k_val) in enumerate(kfold.split(input_array, maccs_fingerprint)):
        k_model = initialize_model(input_size=input_array.shape[1])
        tensorboard_callback_fold = TensorBoard(log_dir=f'./logs/{timestamp}/fold_{i + 1}')
        print(f'--- Training fold {i + 1}')
        k_model.fit(input_array[k_train], maccs_fingerprint[k_train],
                    validation_data=(input_array[k_val], maccs_fingerprint[k_val]), batch_size=32, epochs=1000,
                    callbacks=[early_stopping_callback, reduce_lr_callback, tensorboard_callback_fold], verbose=1)
        scores = k_model.evaluate(input_array[k_val], maccs_fingerprint[k_val], verbose=0)
        print(
            f'Score for fold {i}: {k_model.metrics_names[0]} of {scores[0]}; {k_model.metrics_names[1]} of {scores[1]}')
        metric_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
    print('Hamming Distance CV:')
    print(metric_per_fold)
    print('Loss CV:')
    print(loss_per_fold)
    model.fit(x=input_array, y=maccs_fingerprint, batch_size=32, epochs=1000,
              callbacks=[tensorboard_callback, early_stopping_callback, reduce_lr_callback], validation_split=0.15,
              shuffle=True)
    return model


def predict_model(model: keras.Model, input_array: np.array):
    return model.predict(x=input_array)


def evaluate_model(model: keras.Model, input_array: np.array, maccs_fingerprint: np.array):
    return model.evaluate(x=input_array, y=maccs_fingerprint)


def predict_molecule(model: list):
    """
    Predicts MACCS keys for molecule with probabilities
    :param model: list of XGBoost models which predict the MACCS keys
    :param mol: molecule on which the prediction is performed
    :return: tuple of the form (predicted MACCCS leys, probabilities)
    """
    for m in model:
        proba = m.predict_proba()

