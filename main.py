from model import initialize_model, train_model, predict_model

from database import import_database
from sklearn.model_selection import train_test_split
import numpy as np

nmr_df = import_database()
proton_input_array = np.array(nmr_df['embedded 1H'])
carbon_input_array = np.array(nmr_df['embedded 13C'])
maccs_fingerprint = np.array(nmr_df['MACCS'])

model = initialize_model(max_input_size=100, embedding_length=95, fingerprint_length=166)
print('model initialized')
model = train_model(model=model, carbon_input_array=carbon_input_array, proton_input_array=proton_input_array,
                    maccs_fingerprint=maccs_fingerprint)
print('model trained')

print('Done')
