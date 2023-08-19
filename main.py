import sklearn.metrics

from model import initialize_model, train_model, predict_model, evaluate_model, encode_spectrum, jaccard_index
from database import maccs_to_structure, maccs_to_substructures, import_database, visualize_smarts
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
import numpy as np
import os
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, fbeta_score, roc_curve, auc, \
    make_scorer
from skopt import BayesSearchCV
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
    latent_train, latent_valid, maccs_train, maccs_valid, mol_names_maccs_train, mol_names_maccs_valid = train_test_split(
        latent_train, maccs_train, mol_names_maccs_train, train_size=0.8, shuffle=True)

    clf = []
    y_pred = []
    if not os.path.exists('./saved_xgboost_models'):
        os.mkdir('./saved_xgboost_models')
        with open('metrics_xgboost.csv', 'w+') as f:
            f.write('bit,count,accuracy,precision,recall,f1,f0.5,g-mean,TPR,FPR,AUC,jaccard\n')
            with open(f'./roc.csv', 'w+') as f_roc:
                for bit in range(167):
                    print('bit', bit)
                    scale_pos_bit = np.sqrt(np.count_nonzero([b[bit] == 0 for b in maccs_train]) + 1) / (
                           np.sum([b[bit] for b in maccs_train]) + 1)
                    clf.append(
                        xgboost.XGBClassifier(n_estimators=1000, max_depth=10, subsample=1, eval_metric='auc',
                                              scale_pos_weight=scale_pos_bit, objective='binary:logistic'))
                    clf[-1].fit(latent_train, [b[bit] for b in maccs_train])
                    clf[-1].save_model(f'./saved_xgboost_models/xgboost_{bit}.json')
                    valid_prediction = clf[-1].predict_proba(latent_valid)[:, 1]
                    valid_fpr, valid_tpr, valid_thresholds = roc_curve([b[bit] for b in maccs_valid], valid_prediction,
                                                                       drop_intermediate=False)
                    g_mean = (valid_tpr ** 0.333) * ((1 - valid_fpr) ** 0.666)
                    valid_threshold = valid_thresholds[np.argmax(g_mean)]
                    prediction_prob = clf[-1].predict_proba(latent_test)
                    prediction = []
                    for sample in prediction_prob:
                        if sample[1] > sample[0] and sample[1] > valid_threshold:
                            prediction.append(1)
                        else:
                            prediction.append(0)
                    y_pred.append(prediction)
                    bit_count = np.sum([b[bit] for b in maccs_fingerprint])
                    bit_accuracy = accuracy_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                    bit_precision = precision_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1],
                                                    zero_division=1)
                    bit_recall = recall_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1], zero_division=1)
                    bit_f1 = f1_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1], zero_division=1)
                    bit_f05 = fbeta_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1], beta=0.5,
                                          zero_division=1)
                    best_tpr = valid_tpr[np.argmax(g_mean)]
                    best_fpr = valid_fpr[np.argmax(g_mean)]
                    bit_auc = auc(valid_fpr, valid_tpr)
                    bit_jaccard = jaccard_index(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                    f_roc.write(f'{bit},best_tpr,{best_tpr},tpr,{concatenate_roc(valid_tpr)}\n')
                    f_roc.write(f'{bit},best_fpr,{best_fpr},fpr,{concatenate_roc(valid_fpr)}\n')
                    f.write(f'{bit},{bit_count},{bit_accuracy},{bit_precision},{bit_recall},{bit_f1},{bit_f05},')
                    f.write(f'{np.max(g_mean)},{best_tpr},{best_fpr},{bit_auc},{bit_jaccard}\n')
                    plt.figure(figsize=(5, 5))
                    plt.plot([0, 1], [0, 1], linestyle='--', c='#d8e2dc', zorder=1)
                    plt.plot(valid_fpr, valid_tpr, c='#0fa3b1', zorder=2)
                    plt.scatter([best_fpr], [best_tpr], s=20, c='#f77f00', zorder=3, edgecolors='#9D5100')
                    plt.xlabel('1-Specificity (FPR)')
                    plt.ylabel('Sensitivity (TPR)')
                    plt.title(f'ROC curve for bit {bit}')
                    plt.savefig(f'./roc/bit_{bit}.jpg', dpi=300)
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
