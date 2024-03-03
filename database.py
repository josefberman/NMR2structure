import pickle

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import pandas as pd
import numpy as np
import re
import os
from sklearn.preprocessing import OneHotEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.ordinal import OrdinalEncoder
# from IPython.display import Image, display
import requests
import subprocess

list_of_multiplicities = []
max_intensity = 0


def import_database(dataset_type: str = 'all'):
    """
    Imports the nmrshiftdb2withsignals.sd database by dataset type required as a dataframe
    :param dataset_type: type of dataset: 'all' for all database, 'significant' for MACCS bits over 20, 'CHNO' for \
     molecules containing C,H,N,O atoms. 'carbohydrates' for carbohydrates.
    :return: Structured dataframe from nmrshiftdb2withsignals.sd database
    """
    if not os.path.exists('./database/'):
        os.mkdir('./database/')
    RDLogger.DisableLog('rdApp.*')  # disable RDKit warnings
    nmr_df = read_db_from_pickle()  # try reading stored dataframe if exists
    if nmr_df is None:
        mols = [Chem.AddHs(x) for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd') if x is not None]  # extract
        nmr_df = pd.DataFrame([x.GetPropsAsDict() for x in mols if x is not None])  # create dataframe based on all
        # RDKit molecular and NMR properties
        nmr_df['Molecule'] = mols  # store molecule as RDKit molecule class in dataframe
        nmr_df['Name'] = [x.GetProp('_Name') for x in mols if x is not None]  # store molecule's name in dataframe
        nmr_df['MACCS'] = [maccs_to_list(x) for x in mols if x is not None]  # store MACCS fingerprint in dataframe
        nmr_df['Morgan'] = [mol_to_morgan(x) for x in mols if x is not None]
        nmr_df['Spectrum 13C'] = nmr_df.apply(extract_spectrum, spectrum_type='Spectrum 13C',
                                              axis=1)  # extract first encountered 13C spectrum
        nmr_df['Spectrum 1H'] = nmr_df.apply(extract_spectrum, spectrum_type='Spectrum 1H',
                                             axis=1)  # extract first encountered 1H spectrum
        nmr_df = nmr_df.iloc[:, -6:]  # leave only necessary columns
        print(nmr_df.columns)
        nmr_df = nmr_df[(nmr_df['Spectrum 13C'].notnull()) & (nmr_df['Spectrum 1H'].notnull())]  # leave only molecules
        # with both spectra
        nmr_df['Spectrum 13C'] = nmr_df.apply(extract_smi, spectrum_type='Spectrum 13C',
                                              axis=1)  # convert string format carbon spectrum
        # into lists (input for model)
        nmr_df['Spectrum 1H'] = nmr_df.apply(extract_smi, spectrum_type='Spectrum 1H',
                                             axis=1)  # convert string format proton spectrum
        # into lists (input for model)
        nmr_df['embedded 13C'] = nmr_df.apply(peak_embedding, spectrum_type='Spectrum 13C', axis=1)
        nmr_df['embedded 1H'] = nmr_df.apply(peak_embedding, spectrum_type='Spectrum 1H', axis=1)

        nmr_df = nmr_df[nmr_df.apply(lambda x: len(x['Molecule'].GetAtoms()) <= 37, axis=1)]
        # mols = [x for x in mols if len(x.GetAtoms()) <= 37]
        if dataset_type == 'significant':
            nmr_df = nmr_df[nmr_df.apply(lambda x: is_significant(x['MACCS']), axis=1)]
            # mols = [x for x in mols if]  # limit database to structures only above MACCS bit 20
        elif dataset_type == 'CHNO':
            nmr_df = nmr_df[nmr_df.apply(lambda x: is_CHNO(x['Molecule']), axis=1)]
            # mols = [x for x in mols if is_CHNO(x)]  # limit database to only H,C,O,N
        elif dataset_type == 'carbohydrates':
            nmr_df = nmr_df[nmr_df.apply(lambda x: is_carbohydrate(x['Molecule']), axis=1)]
            # mols = [x for x in mols if is_carbohydrate(x)]  # limit database to only carbohydrates (C,H,O)
        # all molecules with corresponding NMR, and add explicit protons
        print('Length of df:', len(nmr_df))
        write_db_to_pickle(nmr_df)
    return nmr_df


def write_db_to_pickle(nmr_df, pickle_path='./database/nmr_db.pkl'):
    nmr_df.to_pickle(pickle_path)
    with open('./database/global_variables.pkl', 'wb') as file:
        pickle.dump((list_of_multiplicities, max_intensity), file)
    return 0


def read_db_from_pickle(pickle_path='./database/nmr_db.pkl'):
    global list_of_multiplicities
    global max_intensity
    if os.path.exists(pickle_path):
        nmr_df = pd.read_pickle(pickle_path)
        with open('./database/global_variables.pkl', 'rb') as file:
            list_of_multiplicities, max_intensity = pickle.load(file)
        return nmr_df
    return None


def extract_spectrum(nmr_df_row: pd.Series, spectrum_type: str):
    """
    Return the first non-Null carbon spectrum from a dataframe row
    :param spectrum_type: type of spectrum between proton and carbon
    :param nmr_df_row: current molecule's row to extract carbon spectrum
    :return: a proton spectrum (if exists) as string
    """
    temp_row = nmr_df_row.dropna()
    temp_row = temp_row.filter(like=spectrum_type)
    temp_row = temp_row[~temp_row.str.contains('J=')]
    temp_row_with_multiplicity = temp_row[temp_row.str.contains('[A-Za-z]')]
    if temp_row_with_multiplicity.empty:
        if temp_row.empty:
            return None
        elif not temp_row[0]:
            return None
        else:
            return temp_row[0]
    else:
        return temp_row_with_multiplicity[0]


def is_CHNO(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 7, 8]:
            return False
    return True


def is_carbohydrate(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 8]:
            return False
    return True


def is_significant(MACCS_keys: list):
    if sum(MACCS_keys[:41]) > 0:
        return False
    return True


def maccs_to_list(molecule):
    """
    Creates a MACCS fingerprint as a bit array
    :param molecule: the current RDKit molecule
    :return: bit array of the MACCS fingerprint
    """
    fp = MACCSkeys.GenMACCSKeys(molecule).ToBitString()  # extract MACCS fingerprint for molecule and convert
    # to BitString
    return [int(x) for x in fp]


def mol_to_morgan(molecule):
    fpgen = AllChem.GetMorganGenerator(radius=2)
    return list(fpgen.GetFingerprintAsNumPy(molecule))


def maccs_to_structure(maccs_list: list):
    smarts = ''
    idx = [i for i in range(len(maccs_list)) if maccs_list[i] == 1]
    for i in idx:
        smarts = smarts + MACCSkeys.smartsPatts[i][0]
    return smarts


def maccs_to_substructures(maccs_list: list):
    idx = [i for i in range(len(maccs_list)) if maccs_list[i] == 1]
    return [MACCSkeys.smartsPatts[i][0] for i in idx]


def visualize_smarts(dir_path: str, file_name: str, smarts: list):
    for index, smarts_item in enumerate(smarts):
        smarts_item = re.sub(r'%', '%25', smarts_item)
        smarts_item = re.sub(r'&', '%26', smarts_item)
        smarts_item = re.sub(r'\+', '%2B', smarts_item)
        smarts_item = re.sub(r'#', '%23', smarts_item)
        smarts_item = re.sub(r';', '%3B', smarts_item)
        url = 'https://smarts.plus/smartsview/download_rest?smarts=' + smarts_item
        res = requests.get(url)
        with open(f'{dir_path}/{file_name}_sub_{index}.jpg', 'wb') as f:
            f.write(res.content)


def clean_spectra(nmr_df_row: pd.Series):
    nmr_df_row['Spectrum 13C'] = split_peaks(nmr_df_row, 'Spectrum 13C')
    nmr_df_row['Spectrum 1H'] = split_peaks(nmr_df_row, 'Spectrum 1H')
    return nmr_df_row


def split_peaks(nmr_df_row: pd.Series, column_name: str):
    split_str = nmr_df_row[column_name].split('|')
    return split_str


def extract_smi(nmr_df_row: pd.Series, spectrum_type: str):
    """
    Extracting the chemical shift and multiplicity from carbon spectrum
    :param spectrum_type: type of spectrum between proton and carbon
    :param nmr_df_row: row of dataframe containing the carbon spectrum as string
    :return: numpy 2d array of shift,multiplicity
    """
    global list_of_multiplicities
    global max_intensity
    extracted_string = re.findall('([0-9]*.?[0-9]*);[0-9]*.?[0-9]*.?[0-9]*([A-Za-z]+\s?[A-Za-z]*[\^`\']?);[0-9]*\|?',
                                  nmr_df_row[spectrum_type])
    if not extracted_string:
        extracted_string = re.findall(
            '([0-9]*.?[0-9]*);[0-9]*.?[0-9]*.?[0-9]*([A-Za-z]*\s?[A-Za-z]*[\^`\']?);[0-9]*\|?',
            nmr_df_row[spectrum_type])
    extracted_string_new = []
    for element in extracted_string:
        element_as_list = list(element)
        if element_as_list[1] == '':
            element_as_list[1] = 'S'
        extracted_string_new.append(tuple(element_as_list))
    unique_elements, elements_count = np.unique(extracted_string_new, return_counts=True, axis=0)
    return_list = []
    for i, j in zip(unique_elements, elements_count):
        return_list.append([float(i[0]), i[1].lower(), j])
        if i[1].lower() not in list_of_multiplicities:
            list_of_multiplicities.append(i[1].lower())
        if j > max_intensity:
            max_intensity = j
    return return_list


def flatten(list_of_lists):
    result = []
    for item in list_of_lists:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result


def peak_embedding(nmr_df_row: pd.Series, spectrum_type: str):
    # multiplicity_encoder = OneHotEncoder(sparse_output=False, dtype=int)
    intensity_encoder = OneHotEncoder(sparse_output=False, dtype=int)
    # multiplicity_encoder = BinaryEncoder(return_df=False).fit(list_of_multiplicities)
    multiplicity_encoder = OrdinalEncoder(return_df=False).fit(list_of_multiplicities)
    multiplicity_encoder.fit(np.reshape(list_of_multiplicities, (-1, 1)))
    intensity_encoder.fit(np.reshape(range(1, max_intensity + 1), (-1, 1)))
    embedded_row = []
    embedded_element = []
    length_of_embedding = 0
    for element in nmr_df_row[spectrum_type]:
        if spectrum_type == 'Spectrum 13C':
            embedded_element.append(element[0] / 100.0)  # reduce shifts to units (usually hundreds)
        else:
            embedded_element.append(element[0] / 10.0)  # reduce shifts to units (usually tens)
        embedded_element.append(multiplicity_encoder.transform([element[1].lower()]).tolist())
        embedded_element.append(element[2])
        # embedded_element.append(intensity_encoder.transform([[element[2]]]).tolist())
        embedded_row.append(flatten(embedded_element))
        length_of_embedding = len(embedded_row[-1])
        embedded_element = []
    for _ in range(50 - len(nmr_df_row[spectrum_type])):
        embedded_row.append([0] * length_of_embedding)
    return embedded_row
