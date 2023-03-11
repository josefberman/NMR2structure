import keras
import tensorflow
from keras.layers import Dense, Input


def embedding_autoencoder():
    encoder_input = Input(shape=(3,))
    encoder_hidden_layer = Dense(units=64, activation='relu')(encoder_input)
    encoder_output = Dense(units=128, activation='tanh')(encoder_hidden_layer)
    encoder = keras.Model(encoder_input, encoder_output, name='encoder')

    decoder_input = Input(shape=(128,))
    decoder_hidden_layer = Dense(units=65, activation='relu')(decoder_input)
    decoder_output = Dense(units=3, activation='relu')(decoder_hidden_layer)
    decoder = keras.Model(decoder_input, decoder_output, name='decoder')

    autoencoder_input = Input(shape=(3,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = keras.Model(autoencoder_input, decoded, name='autoencoder')

    autoencoder.compile(loss='mse', optimizer='adam')
    return (encoder, decoder, autoencoder)
