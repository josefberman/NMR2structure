from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import pandas as pd
import numpy as np
import re
import os


def import_database():
    """
    Loads a SD file containing molecular properties (including NMR spectra)
    :return: List of the molecules from the SD file
    """
    RDLogger.DisableLog('rdApp.*')  # disable RDKit warnings
    nmr_df = read_db_from_pickle()  # try reading stored dataframe if exists
    if nmr_df is None:
        mols = [Chem.AddHs(x) for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd') if x is not None]  # extract
        # all molecules with corresponding NMR, and add explicit protons
        nmr_df = pd.DataFrame([x.GetPropsAsDict() for x in mols if x is not None])  # create dataframe based on all
        # RDKit molecular and NMR properties
        nmr_df['Molecule'] = mols  # store molecule as RDKit molecule class in dataframe
        nmr_df['Name'] = [x.GetProp('_Name') for x in mols if x is not None]  # store molecule's name in dataframe
        nmr_df['MACCS'] = [maccs_to_list(x) for x in mols if x is not None]  # store MACCS fingerprint in dataframe
        nmr_df['Spectrum 13C'] = nmr_df.apply(extract_spectrum, spectrum_type='Spectrum 13C',
                                              axis=1)  # extract first encountered 13C spectrum
        nmr_df['Spectrum 1H'] = nmr_df.apply(extract_spectrum, spectrum_type='Spectrum 1H',
                                             axis=1)  # extract first encountered 1H spectrum
        nmr_df = nmr_df.iloc[:, -5:]  # leave only necessary columns
        nmr_df = nmr_df[(nmr_df['Spectrum 13C'].notnull()) & (nmr_df['Spectrum 1H'].notnull())]  # leave only molecules
        # with both spectra
        nmr_df['Spectrum 13C'] = nmr_df.apply(extract_smi, spectrum_type='Spectrum 13C',
                                              axis=1)  # convert string format carbon spectrum
        # into lists (input for model)
        nmr_df['Spectrum 1H'] = nmr_df.apply(extract_smi, spectrum_type='Spectrum 1H',
                                             axis=1)  # convert string format proton spectrum
        # into lists (input for model)
        # write_db_to_pickle(nmr_df)
    return nmr_df


def write_db_to_pickle(nmr_df, pickle_path='./database/nmr_db.pkl'):
    nmr_df.to_pickle(pickle_path)
    return 0


def read_db_from_pickle(pickle_path='./database/nmr_db.pkl'):
    if os.path.exists(pickle_path):
        nmr_df = pd.read_pickle(pickle_path)
        return nmr_df
    return None


def extract_spectrum(nmr_df_row: pd.Series, spectrum_type: str):
    """
    Return the first non-Null carbon spectrum from a dataframe row
    :param nmr_df_row: current molecule's row to extract carbon spectrum
    :return: a proton spectrum (if exists) as string
    """
    temp_row = nmr_df_row.dropna()
    temp_row = temp_row.filter(like=spectrum_type)
    temp_row = temp_row[temp_row.str.contains('[A-Za-z]')]
    temp_row = temp_row[~temp_row.str.contains('J=')]
    if temp_row.empty:
        return None
    else:
        return temp_row[0]


def is_carbohydrate(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 8]:
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


def maccs_to_structure(maccs_list: list):
    smarts = ''
    idx = [i for i in range(len(maccs_list)) if maccs_list[i] == 1]
    for i in idx:
        smarts = smarts + MACCSkeys.smartsPatts[i][0]
    return smarts


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
    :param nmr_df_row: row of dataframe containing the carbon spectrum as string
    :return: numpy 2d array of shift,multiplicity
    """
    # extracted_string = re.sub('([0-9]*.?[0-9]*);[0-9]*.?[0-9]*.?[0-9]*([A-Za-z]+\s?[A-Za-z]*[\^`\']?);[0-9]*\|?', '\\1;\\2;',
    #                          nmr_df_row['Spectrum 13C'])
    extracted_string = re.findall('([0-9]*.?[0-9]*);[0-9]*.?[0-9]*.?[0-9]*([A-Za-z]+\s?[A-Za-z]*[\^`\']?);[0-9]*\|?',
                                  nmr_df_row[spectrum_type])
    unique_elements, elements_count = np.unique(extracted_string, return_counts=True, axis=0)
    return_list = []
    for i, j in zip(unique_elements, elements_count):
        return_list.append([float(i[0]), i[1], j])
    return return_list
