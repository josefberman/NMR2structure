import numpy as np
import keras
import tensorflow as tf
from keras.layers import Input, LSTM, Bidirectional, Flatten, Concatenate, Dense
from keras.callbacks import TensorBoard
from datetime import datetime


def initialize_model(max_input_size: int, embedding_length: int, fingerprint_length: int = 166):
    input_layer_carbon = Input(shape=(max_input_size, embedding_length,), name='carbon_input')
    input_layer_proton = Input(shape=(max_input_size, embedding_length,), name='proton_input')
    bilstm_layer_carbon = Bidirectional(LSTM(units=50, activation='relu', return_sequences=True),
                                        name='carbon_BiLSTM')(input_layer_carbon)
    bilstm_layer_proton = Bidirectional(LSTM(units=50, activation='relu', return_sequences=True),
                                        name='proton_BiLSTM')(input_layer_proton)
    flatten_layer_carbon = Flatten(name='flatten_carbon')(bilstm_layer_carbon)
    flatten_layer_proton = Flatten(name='flatten_proton')(bilstm_layer_proton)
    concat_layer = Concatenate(name='concatenate_spectra')([flatten_layer_carbon, flatten_layer_proton])
    output_layer = Dense(units=166, activation='sigmoid', name='fingerprint_output')(concat_layer)
    model = keras.Model(inputs=[input_layer_carbon, input_layer_proton],
                        outputs=output_layer, name='NMR_model')
    model.compile(optimizer='adam', loss='mse', metrics='accuracy')
    return model


def train_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array,
                maccs_fingerprint: np.array):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_callback = TensorBoard(log_dir=f'./logs/{timestamp}', write_graph=True, write_images=True)
    model.fit(x=[carbon_input_array, proton_input_array], y=maccs_fingerprint, batch_size=32, epochs=5,
              callbacks=[tensorboard_callback], validation_split=0.15)
    return model


def predict_model(model: keras.Model, carbon_input_array: np.array, proton_input_array: np.array):
    return model.predict(x=[carbon_input_array, proton_input_array])
