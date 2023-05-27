from model import initialize_model, train_model, predict_model, evaluate_model, encode_spectrum

from database import import_database
from sklearn.model_selection import train_test_split
import numpy as np
from darts.models import NBEATSModel
import os


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    nmr_df = import_database()
    proton_input = np.array(nmr_df['embedded 1H'].tolist())
    carbon_input = np.array(nmr_df['embedded 13C'].tolist())
    print('proton input shape:', proton_input.shape)
    print('carbon input shape:', carbon_input.shape)

    latent_input = encode_spectrum(np.concatenate((proton_input, carbon_input), axis=1))
    print('latent input shape:', latent_input.shape)

    maccs_fingerprint = np.array(nmr_df['MACCS'].tolist())
    print('maccs output shape:', maccs_fingerprint.shape)

    latent_train, latent_test, maccs_train, maccs_test = train_test_split(latent_input, maccs_fingerprint, test_size=0.2)

    model = initialize_model(input_size=latent_train.shape[1], fingerprint_length=167)
    model = train_model(model, latent_train, maccs_train)
    score = evaluate_model(model, latent_test, maccs_test)
    print(f'Eval Huber loss: {score[0]}')
    print(f'Eval Hamming distance: {score[1]*167.0} mismatched bits')

    model.save('saved_model/saved.model.h5')
    model.save('saved_model/')
