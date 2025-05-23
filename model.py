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
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_curve, \
    roc_auc_score, make_scorer, log_loss
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import UndefinedMetricWarning
import csv
from sklearn.model_selection import train_test_split
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.decomposition import PCA


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
    input_train, input_test = train_test_split(input_array, train_size=0.8, shuffle=True, random_state=42)
    # Encoder block
    input_shape = input_train.shape[1:]
    encoder_inputs = Input(shape=(*input_shape, 1))
    x_0 = Flatten(name='flatten_layer_encoder')(encoder_inputs)
    # x_1 = Dense(units=300)(x_0)
    # x_2 = Dense(units=300)(x_1)
    # x_3 = Dense(units=300)(x_2)
    # x_4 = Dense(units=100)(x_3)
    encoder_outputs = Dense(units=50, name='output_layer_encoder')(x_0)

    # Decoder block
    x_4 = Dense(units=300)(encoder_outputs)
    # x_5 = Dense(units=300)(x_4)
    # x_6 = Dense(units=300)(x_5)
    # x_7 = Dense(units=300)(x_6)
    x_8 = Dense(units=x_0.shape[1])(x_4)
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


def predict_bits_from_xgboost(latent_sample):
    with open('./saved_xgboost_models/thresholds.csv', 'r') as thr_f:
        thresholds = list(csv.reader(thr_f, delimiter=','))[0]
    prediction = []
    for bit in range(167):
        if float(thresholds[bit]) == np.inf:
            prediction.append(0)
        elif float(thresholds[bit]) == 0:
            prediction.append(1)
        else:
            try:
                clf = xgboost.XGBClassifier()
                clf.load_model(f'./saved_xgboost_models/xgboost_{bit}.json')
                bit_prediction = clf.predict_proba([latent_sample])[0]
                if bit_prediction[1] > bit_prediction[0] and bit_prediction[1] > float(thresholds[bit]):
                    prediction.append(1)
                else:
                    prediction.append(0)
            except:
                pass

    return prediction


def create_xgboost_model(latent, maccs, tpr_fpr_ratio=0.5):
    # warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    clf_list = []
    if not os.path.exists('./saved_xgboost_models'):
        os.mkdir('./saved_xgboost_models')
    thresholds_list = []
    latent_train, latent_valid, maccs_train, maccs_valid = train_test_split(latent, maccs, test_size=0.2,
                                                                            random_state=42)
    with open('./saved_xgboost_models/best_params.csv', 'w') as f:
        f.write('Bit,max_delta_step,max_depth,n_estimators,subsample,AUC\n')
        for bit in range(167):
            print(f'Bit {bit}')
            prediction = []
            if np.count_nonzero([b[bit] == 0 for b in maccs]) <= 0.01 * np.sum([b[bit] for b in maccs]):
                # very imbalanced classes towards positive class
                prediction.append(1)
                thresholds_list.append(0)
            elif np.sum([b[bit] for b in maccs]) <= 0.01 * np.count_nonzero([b[bit] == 0 for b in maccs]):
                # very imbalanced classes towards negative class
                prediction.append(0)
                thresholds_list.append(np.inf)
            else:
                scale_pos_bit = (np.count_nonzero([b[bit] == 0 for b in maccs]) + 1) / (
                        np.sum([b[bit] for b in maccs]) + 1)
                opt = BayesSearchCV(xgboost.XGBClassifier(scale_pos_weight=scale_pos_bit, objective='binary:logistic',
                                                          eval_metric='auc'),
                                    {'n_estimators': Integer(200, 1000, 'log-uniform'),
                                     'max_depth': Integer(10, 1000, 'log-uniform'),
                                     'subsample': Real(0.6, 1, 'uniform'),
                                     'max_delta_step': Real(0, 100, 'uniform')}, n_iter=100, cv=5,
                                    verbose=0, refit=True, scoring=make_scorer(log_loss, labels=[0, 1]))
                opt.fit(latent_train, [b[bit] for b in maccs_train])
                clf = opt.best_estimator_
                f.write(f'{bit},')
                for element in opt.best_params_.values():
                    f.write(f'{element},')
                f.write(f'{opt.best_score_}\n')
                valid_fpr, valid_tpr, valid_thresholds = roc_curve([b[bit] for b in maccs_valid],
                                                                   clf.predict_proba(latent_valid)[:, 1],
                                                                   pos_label=1, drop_intermediate=False)
                tpr_ratio = tpr_fpr_ratio / (1 + tpr_fpr_ratio)
                fpr_ratio = 1 - tpr_ratio
                g_mean = (valid_tpr ** tpr_ratio) * ((1 - valid_fpr) ** fpr_ratio)
                validation_threshold = valid_thresholds[np.argmax(g_mean)]
                visualize_roc_curve(fpr=valid_fpr, tpr=valid_tpr, thresh_fpr=valid_fpr[np.argmax(g_mean)],
                                    thresh_tpr=valid_tpr[np.argmax(g_mean)], bit=bit)
                validation_prediction = clf.predict_proba(latent_valid)
                for sample in validation_prediction:
                    if sample[1] > sample[0] and sample[1] > validation_threshold:
                        prediction.append(1)
                    else:
                        prediction.append(0)
                thresholds_list.append(validation_threshold)
    best_params = pd.read_csv('./saved_xgboost_models/best_params.csv', header=0,
                              dtype={'Bit': 'int64', 'max_delta_step': 'float64', 'n_estimators': 'int64',
                                     'subsample': 'float64', 'AUC': 'float64'}, index_col='Bit')
    for bit in best_params.index:
        scale_pos_bit = (np.count_nonzero([b[bit] == 0 for b in maccs]) + 1) / (
                np.sum([b[bit] for b in maccs]) + 1)
        clf_list.append(
            xgboost.XGBClassifier(n_estimators=best_params.loc[bit, 'n_estimators'],
                                  max_depth=best_params.loc[bit, 'max_depth'],
                                  subsample=best_params.loc[bit, 'subsample'], eval_metric='auc',
                                  scale_pos_weight=scale_pos_bit, objective='binary:logistic',
                                  max_delta_step=best_params.loc[bit, 'max_delta_step']))
        clf_list[-1].fit(latent, [b[bit] for b in maccs])
        clf_list[-1].save_model(f'./saved_xgboost_models/xgboost_{bit}.json')
    with open('./saved_xgboost_models/thresholds.csv', 'w') as thr_f:
        for threshold in thresholds_list:
            thr_f.write(f'{threshold},')
    return clf_list


def cv_xgboost_model(latent, maccs, tpr_fpr_ratio=0.5):
    if not os.path.exists('./saved_xgboost_models'):
        os.mkdir('./saved_xgboost_models')
    # warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
    fbeta_df = pd.DataFrame()
    auc_df = pd.DataFrame()
    g_mean_df = pd.DataFrame()
    precision_df = pd.DataFrame()
    recall_df = pd.DataFrame()
    thresholds_list = []
    fbeta_scores = {}
    auc_scores = {}
    g_mean_scores = {}
    precision_scores = {}
    recall_scores = {}
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    for split_index, (train_i, valid_i) in enumerate(kf.split(latent)):
        for bit in range(167):
            print(f'Fold {split_index}: Bit {bit}')
            prediction = []
            if np.count_nonzero([b[bit] == 0 for b in maccs[train_i]]) <= 0.01 * np.sum(
                    [b[bit] for b in maccs[train_i]]):
                # very imbalanced classes towards positive class
                prediction.append(1)
                thresholds_list.append(0)
                auc_scores[bit] = 1
                precision_scores[bit] = 1
                recall_scores[bit] = 1
                fbeta_scores[bit] = 1
                g_mean_scores[bit] = 1
            elif np.sum([b[bit] for b in maccs[train_i]]) <= 0.01 * np.count_nonzero(
                    [b[bit] == 0 for b in maccs[train_i]]):
                # very imbalanced classes towards negative class
                prediction.append(0)
                thresholds_list.append(np.inf)
                auc_scores[bit] = 1
                precision_scores[bit] = 1
                recall_scores[bit] = 1
                fbeta_scores[bit] = 1
                g_mean_scores[bit] = 1
            else:
                scale_pos_bit = (np.count_nonzero([b[bit] == 0 for b in maccs[train_i]]) + 1) / (
                        np.sum([b[bit] for b in maccs[train_i]]) + 1)
                opt = BayesSearchCV(xgboost.XGBClassifier(scale_pos_weight=scale_pos_bit, objective='binary:logistic',
                                                          eval_metric='auc'),
                                    {'n_estimators': Integer(200, 1000, 'log-uniform'),
                                     'max_depth': Integer(10, 1000, 'log-uniform'),
                                     'subsample': Real(0.6, 1, 'uniform'),
                                     'max_delta_step': Real(0, 100, 'uniform')}, n_iter=20, cv=3,
                                    verbose=0, refit=True, scoring=make_scorer(log_loss, labels=[0, 1]))
                opt.fit(latent[train_i], [b[bit] for b in maccs[train_i]])
                clf = opt.best_estimator_
                valid_fpr, valid_tpr, valid_thresholds = roc_curve([b[bit] for b in maccs[valid_i]],
                                                                   clf.predict_proba(latent[valid_i])[:, 1],
                                                                   pos_label=1, drop_intermediate=False)
                tpr_ratio = tpr_fpr_ratio / (1 + tpr_fpr_ratio)
                fpr_ratio = 1 - tpr_ratio
                g_mean = (valid_tpr ** tpr_ratio) * ((1 - valid_fpr) ** fpr_ratio)
                validation_threshold = valid_thresholds[np.argmax(g_mean)]
                validation_prediction = clf.predict_proba(latent[valid_i])
                for sample in validation_prediction:
                    if sample[1] > sample[0] and sample[1] > validation_threshold:
                        prediction.append(1)
                    else:
                        prediction.append(0)
                fbeta_scores[bit] = fbeta_score([b[bit] for b in maccs[valid_i]], prediction, beta=0.5,
                                                zero_division=1.0)
                try:
                    # auc_scores[bit] = roc_auc_score([b[bit] for b in maccs_valid],
                    #                                clf.predict_proba(latent_valid)[:, 1])
                    auc_scores[bit] = roc_auc_score([b[bit] for b in maccs[valid_i]], prediction)
                except ValueError:
                    auc_scores[bit] = 1.0
                if np.isnan(np.max(g_mean)):
                    g_mean_scores[bit] = 1.0
                else:
                    g_mean_scores[bit] = np.max(g_mean)
                precision_scores[bit] = precision_score([b[bit] for b in maccs[valid_i]], prediction,
                                                        zero_division=1.0)
                recall_scores[bit] = recall_score([b[bit] for b in maccs[valid_i]], prediction, zero_division=1.0)
        fbeta_df[f'split {split_index}'] = fbeta_scores
        auc_df[f'split {split_index}'] = auc_scores
        g_mean_df[f'split {split_index}'] = g_mean_scores
        precision_df[f'split {split_index}'] = precision_scores
        recall_df[f'split {split_index}'] = recall_scores
    with pd.ExcelWriter('./saved_xgboost_models/scores.xlsx', engine='openpyxl', mode='w') as writer:
        fbeta_df.to_excel(writer, sheet_name='fbeta', header=True, index=True)
    with pd.ExcelWriter('./saved_xgboost_models/scores.xlsx', engine='openpyxl', mode='a') as writer:
        auc_df.to_excel(writer, sheet_name='auc', header=True, index=True)
        g_mean_df.to_excel(writer, sheet_name='g_mean', header=True, index=True)
        precision_df.to_excel(writer, sheet_name='precision', header=True, index=True)
        recall_df.to_excel(writer, sheet_name='recall', header=True, index=True)
    return None


def visualize_roc_curve(fpr, tpr, thresh_fpr, thresh_tpr, bit: int):
    if not os.path.exists('./roc/'):
        os.mkdir('./roc/')
    f = plt.figure(figsize=(10, 10))
    plt.plot([0, 1], [0, 1], c='#bfc0c0', linestyle='--', zorder=1, linewidth=2)
    plt.plot(fpr, tpr, c='#669BBC', zorder=5, linewidth=4, markersize=200, markerfacecolor='k')
    plt.scatter(x=[thresh_fpr], y=[thresh_tpr], s=200, c='#C1121F', zorder=10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    f.savefig(f'./roc/roc_bit_{bit}.jpg', dpi=200)
    f.clear()
    plt.close(f)
