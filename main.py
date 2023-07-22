import sklearn.metrics

from model import initialize_model, train_model, predict_model, evaluate_model, encode_spectrum
from database import maccs_to_structure, maccs_to_substructures, import_database, visualize_smarts
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold, GridSearchCV
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import xgboost

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    nmr_df = import_database()
    proton_input = np.array(nmr_df['embedded 1H'].tolist())
    carbon_input = np.array(nmr_df['embedded 13C'].tolist())
    proton_input = proton_input / np.max(proton_input)
    carbon_input = carbon_input / np.max(carbon_input)
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
        latent_input, maccs_fingerprint, mol_names_maccs, train_size=0.8, shuffle=True)

    clf = []
    y_pred = []
    if not os.path.exists('./saved_xgboost_models'):
        os.mkdir('./saved_xgboost_models')
        with open('metrics_xgboost.csv', 'w+') as f:
            f.write('bit,weight,accuracy,f1\n')
            for bit in range(167):
                model = xgboost.XGBClassifier(n_estimators=100, max_depth=3)
                weights = [0.1, 1, 10, 100, 1_000, 10_000]
                param_grid = dict(scale_pos_weight=weights)
                cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=1)
                grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='f1')
                grid_result = grid.fit(latent_train, [b[bit] for b in maccs_train])
                print(f'Best for bit {bit}: {grid_result.best_score_} using {list(grid_result.best_params_.values())[0]}')
                clf.append(xgboost.XGBClassifier(n_estimators=100, max_depth=3,
                                                 scale_pos_weight=list(grid_result.best_params_.values())[0]))
                clf[-1].fit(latent_train, [b[bit] for b in maccs_train])
                clf[-1].save_model(f'./saved_xgboost_models/xgboost_{bit}.json')
                y_pred.append(clf[-1].predict(latent_test))
                bit_accuracy = accuracy_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                bit_f1 = f1_score(y_true=[b[bit] for b in maccs_test], y_pred=y_pred[-1])
                print(f'accuracy for bit {bit} is {bit_accuracy}')
                print(f'f1 for bit {bit} is {bit_f1}')
                f.write(f'{bit},{list(grid_result.best_params_.values())[0]},{bit_accuracy},{bit_f1}\n')
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
