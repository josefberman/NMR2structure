from model import encode_spectrum, cv_xgboost_model, create_xgboost_model, predict_bits_from_xgboost
from database import import_database, visualize_smarts, maccs_to_substructures
from sklearn.model_selection import train_test_split
import numpy as np
import os
import warnings


def concatenate_roc(l: list):
    return ','.join([str(x) for x in l])


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    warnings.filterwarnings('ignore')

    nmr_df = import_database('all')
    proton_input = np.array(nmr_df['embedded 1H'].tolist())
    carbon_input = np.array(nmr_df['embedded 13C'].tolist())

    mol_names_maccs = nmr_df.loc[:, ['Name', 'MACCS']].reset_index(drop=True)
    print('max proton ', np.max(proton_input))
    print('max carbon ', np.max(carbon_input))

    print('proton input shape:', proton_input.shape)
    print('carbon input shape:', carbon_input.shape)

    # latent_input = encode_spectrum(np.concatenate((proton_input, carbon_input), axis=1))

    latent_input = np.concatenate((proton_input, carbon_input), axis=1)
    latent_input = latent_input.reshape(latent_input.shape[0], -1)

    maccs_fingerprint = np.array(nmr_df['MACCS'].tolist())

    print('latent input shape:', latent_input.shape)
    print('maccs output shape:', maccs_fingerprint.shape)

    latent_train, latent_test, maccs_train, maccs_test, mol_names_maccs_train, mol_names_maccs_test = train_test_split(
        latent_input, maccs_fingerprint, mol_names_maccs, train_size=0.90, shuffle=True, random_state=42)
    cv_xgboost_model(latent=latent_train, maccs=maccs_train, tpr_fpr_ratio=0.25)
    create_xgboost_model(latent=latent_train, maccs=maccs_train, tpr_fpr_ratio=0.25)

    tested_molecules = 30
    predicted_test = []
    for index, item in enumerate(latent_test):
        if index < tested_molecules:
            predicted_test.append(predict_bits_from_xgboost(item))
        else:
            predicted_test.append(None)
    mol_names_maccs_test['predicted'] = predicted_test
    mol_names_maccs_test.reset_index(inplace=True, drop=True)

    for index, item in mol_names_maccs_test[:tested_molecules].iterrows():
        os.makedirs(name=f'./test/test_{index}/ground_truth/', exist_ok=True)
        gt_smarts = maccs_to_substructures(item['MACCS'])
        print(f"Molecule {index}, {sum(item['MACCS'])} substructures found in ground truth")
        visualize_smarts(dir_path=f'./test/test_{index}/ground_truth', file_name=f'mol_{index}', smarts=gt_smarts)
        with open(f'./test/test_{index}/ground_truth/mol_{index}_name.csv', 'a') as f:
            f.write(f"Molecule {index} name:{item['Name']}\n")
        os.makedirs(name=f'./test/test_{index}/predicted/', exist_ok=True)
        predicted_smarts = maccs_to_substructures(item['predicted'])
        print(f"Molecule {index}, {sum(item['predicted'])} substructures found in prediction")
        visualize_smarts(dir_path=f'./test./test_{index}/predicted', file_name=f'mol_{index}', smarts=predicted_smarts)
        with open(f'./test/test_{index}/predicted/mol_{index}_name.csv', 'a') as f:
            f.write(f"Molecule {index} name:{item['Name']}\n")
    print('done')
