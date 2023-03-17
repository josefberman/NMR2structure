from database import import_database
from sklearn.model_selection import train_test_split
import numpy as np

nmr_df = import_database()
proton_input_array = np.array(nmr_df['Spectrum 1H'])
carbon_input_array = np.array(nmr_df['Spectrum 13C'])


print('Done')
