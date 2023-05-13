from model import initialize_model, train_model, predict_model, evaluate_model, encode_spectrum

from database import import_database
from sklearn.model_selection import train_test_split
import numpy as np
from darts.models import NBEATSModel
import os

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
print(f'Eval loss: {score[0]}\nEval jacard index: {score[1]}\nEval Hamming distance: {score[2]*167.0}')

# unified_input_train, unified_input_test, maccs_fingerprint_train, maccs_fingerprint_test = train_test_split(
#     unified_input_array, maccs_fingerprint, test_size=0.15)
# print('Input train shape:', unified_input_train.shape)
# print('Output train shape:', maccs_fingerprint_train.shape)
# print('Input test shape:', unified_input_test.shape)
# print('Output test shape:', maccs_fingerprint_test.shape)


# METHOD = 'train'  # hp/train
# if METHOD == 'train':
#     model = initialize_model(input_size=200, embedding_length=56, fingerprint_length=167)
#     print('model initialized')
#     model = train_model(model=model, carbon_input_array=carbon_input_train, proton_input_array=proton_input_train,
#                         maccs_fingerprint=maccs_fingerprint_train)
#     print('model trained')
#     score = evaluate_model(model=model, carbon_input_array=carbon_input_test, proton_input_array=proton_input_test,
#                            maccs_fingerprint=maccs_fingerprint_test)
#     print(f'Eval loss: {score[0]}\nEval accuracy: {score[1]}')
#
# else:
#     model = train_model_with_hp_tuning(carbon_input_train, proton_input_train, maccs_fingerprint_train)
#     score = evaluate_model(model=model, carbon_input_array=carbon_input_test, proton_input_array=proton_input_test,
#                            maccs_fingerprint=maccs_fingerprint_test)
#
#
# print('Done')
