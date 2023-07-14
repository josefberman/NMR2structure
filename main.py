import sklearn.metrics

from model import initialize_model, train_model, predict_model, evaluate_model, encode_spectrum
from database import maccs_to_structure, maccs_to_substructures, import_database, visualize_smarts
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
        latent_input, maccs_fingerprint, mol_names_maccs, train_size=0.8)

    model = initialize_model(input_size=latent_train.shape[1], fingerprint_length=167)
    model = train_model(model, latent_train, maccs_train)
    score = evaluate_model(model, latent_test, maccs_test)
    print(f'Eval {model.metrics_names[0]}: {score[0]}')
    print(f'Eval {model.metrics_names[1]}: {score[1]}')

    predicted_maccs = predict_model(model, latent_test)
    """
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
    """
    round_maccs = []
    for maccs in predicted_maccs:
        round_maccs.append(maccs.round())
    y_true = np.array(mol_names_maccs_test['MACCS'].tolist()).transpose()
    y_pred = np.array(round_maccs).transpose()
    """
    print('[', end='')
    for i in range(len(y_true)):
        print(np.round(accuracy_score(y_true[i], y_pred[i]), 3), end=',')
    print(']')
    print('[', end='')
    for i in range(len(y_true)):
        print(np.round(recall_score(y_true[i], y_pred[i]), 3), end=',')
    print(']')
    print('[', end='')
    for i in range(len(y_true)):
        print(np.round(precision_score(y_true[i], y_pred[i]), 3), end=',')
    print(']')
    """
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
