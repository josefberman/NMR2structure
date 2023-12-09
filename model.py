import keras
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model
from keras.optimizers import Adam
import os
import xgboost
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_curve, \
    roc_auc_score
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import csv


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


def create_xgboost_model(latent_train, maccs_train, tpr_fpr_ratio=0.5):
    clf = []
    if not os.path.exists('./saved_xgboost_models'):
        os.mkdir('./saved_xgboost_models')
        with open('./saved_xgboost_models/thresholds.csv', 'w+') as thr_f:
            for bit in range(167):
                scale_pos_bit = np.sqrt(np.count_nonzero([b[bit] == 0 for b in maccs_train]) + 1) / (
                        np.sum([b[bit] for b in maccs_train]) + 1)
                clf.append(
                    xgboost.XGBClassifier(n_estimators=200, max_depth=10, subsample=1, eval_metric='auc',
                                          scale_pos_weight=scale_pos_bit, objective='binary:logistic'))
                clf[-1].fit(latent_train, [b[bit] for b in maccs_train])
                clf[-1].save_model(f'./saved_xgboost_models/xgboost_{bit}.json')
                fpr, tpr, thresholds = roc_curve([b[bit] == 0 for b in maccs_train],
                                                 clf[-1].predict_proba(latent_train)[:, 1], drop_intermediate=False)
                tpr_ratio = tpr_fpr_ratio / (1 + tpr_fpr_ratio)
                fpr_ratio = 1 - tpr_ratio
                g_mean = (tpr ** tpr_ratio) * ((1 - fpr) ** fpr_ratio)
                threshold = thresholds[np.argmax(g_mean)]
                thr_f.write(f'{threshold},')
    return clf


def predict_bits_from_xgboost(latent_sample):
    if os.path.exists('./saved_xgboost_models'):
        with open('./saved_xgboost_models/thresholds.csv', 'r') as thr_f:
            thresholds = list(csv.reader(thr_f, delimiter=','))[0]
            print(len(thresholds))
        prediction = []
        for bit in range(167):
            clf = xgboost.XGBClassifier()
            clf.load_model(f'./saved_xgboost_models/xgboost_{bit}.json')
            bit_prediction = clf.predict_proba([latent_sample])[0]
            if bit_prediction[1] > bit_prediction[0] and bit_prediction[1] > float(thresholds[bit]):
                prediction.append(1)
            else:
                prediction.append(0)
        return prediction
    return None


def cv_xgboost_model(num_cv_folds, latent_train, maccs_train, tpr_fpr_ratio=0.5):
    warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    cv_folds = KFold(n_splits=num_cv_folds)
    fbeta_df = pd.DataFrame()
    auc_df = pd.DataFrame()
    g_mean_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    for i, (train_index, test_index) in enumerate(cv_folds.split(latent_train, maccs_train)):
        print(f'Fold {i}')
        fbeta_scores = {}
        auc_scores = {}
        g_mean_scores = {}
        precision_scores = {}
        recall_scores = {}
        for bit in range(167):
            scale_pos_bit = np.sqrt(np.count_nonzero([b[bit] == 0 for b in maccs_train[train_index]]) + 1) / (
                    np.sum([b[bit] for b in maccs_train[train_index]]) + 1)
            clf = xgboost.XGBClassifier(n_estimators=200, max_depth=10, subsample=1, eval_metric='auc',
                                        scale_pos_weight=scale_pos_bit, objective='binary:logistic')
            clf.fit(latent_train[train_index], [b[bit] for b in maccs_train[train_index]])
            valid_prediction = clf.predict_proba(latent_train[test_index])[:, 1]
            valid_fpr, valid_tpr, valid_thresholds = roc_curve([b[bit] for b in maccs_train[test_index]],
                                                               valid_prediction, drop_intermediate=False)
            tpr_ratio = tpr_fpr_ratio / (1 + tpr_fpr_ratio)
            fpr_ratio = 1 - tpr_ratio
            g_mean = (valid_tpr ** tpr_ratio) * ((1 - valid_fpr) ** fpr_ratio)
            validation_threshold = valid_thresholds[np.argmax(g_mean)]
            visualize_roc_curve(fpr=valid_fpr, tpr=valid_tpr, thresh_fpr=valid_fpr[np.argmax(g_mean)],
                                thresh_tpr=valid_tpr[np.argmax(g_mean)], bit=bit, fold=i)
            validation_prediction = clf.predict_proba(latent_train[test_index])
            prediction = []
            for sample in validation_prediction:
                if sample[1] > sample[0] and sample[1] > validation_threshold:
                    prediction.append(1)
                else:
                    prediction.append(0)
            fbeta_scores[bit] = fbeta_score([b[bit] for b in maccs_train[test_index]], prediction, beta=0.5,
                                            zero_division=1.0)
            try:
                auc_scores[bit] = roc_auc_score([b[bit] for b in maccs_train[test_index]], valid_prediction)
            except ValueError:
                auc_scores[bit] = 1.0
            if np.isnan(np.max(g_mean)):
                g_mean_scores[bit] = 1.0
            else:
                g_mean_scores[bit] = np.max(g_mean)
            precision_scores[bit] = precision_score([b[bit] for b in maccs_train[test_index]], prediction,
                                                    zero_division=1.0)
            recall_scores[bit] = recall_score([b[bit] for b in maccs_train[test_index]], prediction,
                                              zero_division=1.0)
        fbeta_df[f'fold_{i}'] = fbeta_scores
        auc_df[f'fold_{i}'] = auc_scores
        g_mean_df[f'fold_{i}'] = g_mean_scores
        precision_df[f'fold_{i}'] = precision_scores
        recall_df[f'fold_{i}'] = recall_scores
    with pd.ExcelWriter('./cv_scores.xlsx', engine='openpyxl', mode='w') as writer:
        fbeta_df.to_excel(writer, sheet_name='fbeta', header=True, index=True)
    with pd.ExcelWriter('./cv_scores.xlsx', engine='openpyxl', mode='a') as writer:
        auc_df.to_excel(writer, sheet_name='auc', header=True, index=True)
        g_mean_df.to_excel(writer, sheet_name='g_mean', header=True, index=True)
        precision_df.to_excel(writer, sheet_name='precision', header=True, index=True)
        recall_df.to_excel(writer, sheet_name='recall', header=True, index=True)


def visualize_roc_curve(fpr, tpr, thresh_fpr, thresh_tpr, fold: int, bit: int):
    if not os.path.exists('./new_roc/'):
        os.mkdir('./new_roc/')
    f = plt.figure(figsize=(15, 15))
    plt.plot([0, 1], [0, 1], c='#bfc0c0', linestyle='--', zorder=1, linewidth=2)
    plt.plot(fpr, tpr, c='#3a6ea5', zorder=5, linewidth=2)
    plt.scatter(x=[thresh_fpr], y=[thresh_tpr], s=40, c='#d77a61', zorder=10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    f.savefig(f'./new_roc/roc_bit_{bit}_fold_{fold}.jpg', dpi=200)
    f.clear()
    plt.close(f)
