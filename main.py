import sklearn.metrics

from model import initialize_model, train_model, predict_model, evaluate_model, encode_spectrum
from database import maccs_to_structure, maccs_to_substructures, import_database, visualize_smarts
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_curve, auc, \
    make_scorer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost
import matplotlib.pyplot as plt


def concatenate_roc(l: list):
    return ','.join([str(x) for x in l])


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    nmr_df = import_database()
    proton_input = np.array(nmr_df['embedded 1H'].tolist())
    carbon_input = np.array(nmr_df['embedded 13C'].tolist())
    # proton_input = proton_input / np.max(proton_input)
    # carbon_input = carbon_input / np.max(carbon_input)
    mol_names_maccs = nmr_df.loc[:, ['Name', 'MACCS']]
    print('max proton ', np.max(proton_input))
    print('max carbon ', np.max(carbon_input))

    print('proton input shape:', proton_input.shape)
    print('carbon input shape:', carbon_input.shape)

    latent_input = encode_spectrum(np.concatenate((proton_input, carbon_input), axis=1))
    print('latent input shape:', latent_input.shape)

    maccs_fingerprint = np.array(nmr_df['MACCS'].tolist())
    print('maccs output shape:', maccs_fingerprint.shape)

    latent_train, latent_test, maccs_train, maccs_test, mol_names_maccs_train, mol_names_maccs_test = train_test_split(
        latent_input, maccs_fingerprint, mol_names_maccs, train_size=0.95, shuffle=True)

    clf = []
    y_pred = []
    if not os.path.exists('./saved_xgboost_models'):
        os.mkdir('./saved_xgboost_models')
        with open('metrics_xgboost.csv', 'w+') as f:
            f.write('bit,count,accuracy,precision,recall,f1,f0.5,g-mean,TPR,FPR,AUC\n')
            with open(f'./roc.csv', 'w+') as f_roc:
                for bit in range(167):
                    print('bit', bit)
                    scale_pos_bit = np.sqrt(np.sum([b[bit] == 0 for b in maccs_train]) + 1) / (np.sum(
                        [b[bit] for b in maccs_train]) + 1)
                    bit_model = xgboost.XGBClassifier(scale_pos_weight=scale_pos_bit, objective='binary:logistic',
                                                      eval_metric='logloss')
                    bit_params = [{'n_estimators': [200, 500,1000], 'subsample': [0.7, 0.8, 0.9],
                                   'max_depth': [3, 5, 8, 15, 20], 'base_score': [0.1, 0.3, 0.5, 0.7, 0.9]}]
                    bit_scorer = make_scorer(fbeta_score, beta=0.25, zero_division=0.0)
                    bit_gridsearch = GridSearchCV(bit_model, param_grid=bit_params, scoring=bit_scorer, cv=5)
                    bit_gridsearch.fit(latent_train, [b[bit] for b in maccs_train])
                    print(f'AUC for bit {bit}: {bit_gridsearch.best_score_}')
                    clf.append(xgboost.XGBClassifier(n_estimators=bit_gridsearch.best_params_['n_estimators'],
                                                     max_depth=bit_gridsearch.best_params_['max_depth'],
                                                     scale_pos_weight=scale_pos_bit,
                                                     subsample=bit_gridsearch.best_params_['subsample'],
                                                     objective='binary:logistic', eval_metric='logloss'))
                    clf[-1].fit(latent_train, [b[bit] for b in maccs_train])
                    clf[-1].save_model(f'./saved_xgboost_models/xgboost_{bit}.json')
                    train_prediction = clf[-1].predict_proba(latent_train)[:, 1]
                    train_tpr, train_fpr, train_thresholds = roc_curve([b[bit] for b in maccs_train], train_prediction,
                                                                       drop_intermediate=False)
                    g_mean = (train_tpr ** 0.25) * ((1 - train_fpr) ** 0.75)
                    train_threshold = train_thresholds[np.argmax(g_mean)]
                    prediction_prob = clf[-1].predict_proba(latent_test)[:, 1]
                    prediction = [p > train_threshold for p in prediction_prob]
                    y_pred.append(prediction)
                    bit_count = np.sum([b[bit] for b in maccs_fingerprint])
                    bit_accuracy = accuracy_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                    bit_precision = precision_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                    bit_recall = recall_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                    bit_f1 = f1_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                    bit_f05 = fbeta_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1], beta=0.5)
                    bit_tpr, bit_fpr, _ = roc_curve([b[bit] for b in maccs_test], prediction_prob,
                                                    drop_intermediate=False)
                    g_mean = (bit_tpr ** 0.25) * ((1 - bit_fpr) ** 0.75)
                    best_tpr = bit_tpr[np.argmax(g_mean)]
                    best_fpr = bit_fpr[np.argmax(g_mean)]
                    bit_auc = auc(bit_fpr, bit_tpr)
                    f_roc.write(f'{bit},best_tpr,{best_tpr},tpr,{concatenate_roc(bit_tpr)}\n')
                    f_roc.write(f'{bit},best_fpr,{best_fpr},fpr,{concatenate_roc(bit_fpr)}\n')
                    f.write(f'{bit},{bit_count},{bit_accuracy},{bit_precision},{bit_recall},{bit_f1},{bit_f05},')
                    f.write(f'{np.max(g_mean)},{best_tpr},{best_fpr},{bit_auc}\n')
    else:
        for bit in range(167):
            model = xgboost.XGBClassifier()
            model.load_model(f'./saved_xgboost_models/xgboost_{bit}.json')
            clf.append(model)
            y_pred.append(clf[-1].predict(latent_test))
    y_pred_array = np.array(y_pred).reshape(167, -1).T
    print('pred shape:', y_pred_array.shape)
    for i in range(5):
        print('test:')
        print(mol_names_maccs_test['Name'].iloc[i])
        print(maccs_to_substructures(mol_names_maccs_test['MACCS'].iloc[i]))
        for smarts_index, smarts in enumerate(maccs_to_substructures(mol_names_maccs_test['MACCS'].iloc[i])):
            visualize_smarts('gt', i, smarts_index, smarts)
        print('predicted:')
        print(maccs_to_substructures(y_pred_array[i]))
        for smarts_index, smarts in enumerate(maccs_to_substructures(y_pred_array[i])):
            visualize_smarts('pred', i, smarts_index, smarts)
        print('\n')
    print('done')

"""
    with open('maccs.csv', 'w+') as f:
        for maccs in maccs_fingerprint:
            for i in maccs:
                f.write(f'{i},')
            f.write('\n')

    model = initialize_model(input_size=latent_train.shape[1], fingerprint_length=167)
    model = train_model(model, latent_train, maccs_train)
    score = evaluate_model(model, latent_test, maccs_test)
    print(f'Eval {model.metrics_names[0]}: {score[0]}')
    print(f'Eval {model.metrics_names[1]}: {score[1]}')

    predicted_maccs = predict_model(model, latent_test)

    # substructure visualization
    for mol_index, maccs in enumerate(predicted_maccs[:5]):
        print(mol_names_maccs_test['Name'].iloc[mol_index])
        print(maccs_to_substructures(mol_names_maccs_test['MACCS'].iloc[mol_index]))
        for smarts_index, smarts in enumerate(maccs_to_substructures(mol_names_maccs_test['MACCS'].iloc[mol_index])):
            visualize_smarts('gt', mol_index, smarts_index, smarts)
        round_maccs = maccs.round()
        print('Predicted SMARTS:', end=' ')
        print(maccs_to_substructures(round_maccs))
        for smarts_index, smarts in enumerate(maccs_to_substructures(round_maccs)):
            visualize_smarts('pred', mol_index, smarts_index, smarts)
        print('\n')

    # metrics report
    round_maccs = []
    for maccs in predicted_maccs:
        round_maccs.append(maccs.round())
    y_true = np.array(mol_names_maccs_test['MACCS'].tolist()).transpose()
    y_pred = np.array(round_maccs).transpose()
    with open('./metrics/metrics.csv', 'w+') as f:
        f.write(f'Count,Accuracy,Precision,Recall,f1\n')
        for i in range(len(y_true)):
            f.write(f'{i},{sum(y_true[i])},'
                    f'{np.round(accuracy_score(y_true[i], y_pred[i]), 3)},'
                    f'{np.round(precision_score(y_true[i], y_pred[i]), 3)},'
                    f'{np.round(recall_score(y_true[i], y_pred[i]), 3)},'
                    f'{np.round(f1_score(y_true[i], y_pred[i]), 3)}\n')
    # print('[', end='')
    # for i in range(len(y_true)):
    #     print(sum(y_true[i]), end=',')
    # print(']')

    model.save('saved_model/saved.model.h5')
    model.save('saved_model/')
"""
