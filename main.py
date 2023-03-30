from model import initialize_model, train_model, predict_model, evaluate_model

from database import import_database
from sklearn.model_selection import train_test_split
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

nmr_df = import_database()
proton_input_array = np.array(nmr_df['embedded 1H'].tolist())
carbon_input_array = np.array(nmr_df['embedded 13C'].tolist())
maccs_fingerprint = np.array(nmr_df['MACCS'].tolist())
proton_input_train, proton_input_test, carbon_input_train, carbon_input_test, \
    maccs_fingerprint_train, maccs_fingerprint_test = train_test_split(proton_input_array, carbon_input_array,
                                                                       maccs_fingerprint, test_size=0.15)

model = initialize_model(input_size=200, embedding_length=95, fingerprint_length=167)
print('model initialized')
model = train_model(model=model, carbon_input_array=carbon_input_train, proton_input_array=proton_input_train,
                    maccs_fingerprint=maccs_fingerprint_train)
print('model trained')

score = evaluate_model(model=model, carbon_input_array=carbon_input_test, proton_input_array=proton_input_test,
                       maccs_fingerprint=maccs_fingerprint_test)

print(f'Eval loss: {score[0]}\nEval accuracy: {score[1]}')

print('Done')
