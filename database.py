from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import pandas as pd
import os


def import_database():
    """
    Loads a SD file containing molecular properties (including NMR spectra)
    :return: List of the molecules from the SD file
    """
    RDLogger.DisableLog('rdApp.*')
    nmr_df = read_db_from_pickle()
    if nmr_df is None:
        mols = [Chem.AddHs(x) for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd') if x is not None]
        # mols = [mol for mol in mols if is_carbohydrate(mol)]
        nmr_df = pd.DataFrame([x.GetPropsAsDict() for x in mols if x is not None])
        nmr_df['Molecule'] = mols
        nmr_df['Name'] = [x.GetProp('_Name') for x in mols if x is not None]
        nmr_df['MACCS'] = [maccs_to_list(x) for x in mols if x is not None]
        nmr_df['Spectrum 13C'] = nmr_df.apply(extract_carbon_spectrum, axis=1)
        nmr_df['Spectrum 1H'] = nmr_df.apply(extract_proton_spectrum, axis=1)
        nmr_df = nmr_df[(nmr_df['Spectrum 13C'].notnull()) & (nmr_df['Spectrum 1H'].notnull())]
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


def extract_carbon_spectrum(nmr_df_row: pd.Series):
    temp_row = nmr_df_row.dropna()
    temp_row = temp_row.filter(like='Spectrum 13C')
    if temp_row.empty:
        return None
    else:
        return temp_row[0]


def extract_proton_spectrum(nmr_df_row: pd.Series):
    temp_row = nmr_df_row.dropna()
    temp_row = temp_row.filter(like='Spectrum 1H')
    if temp_row.empty:
        return None
    else:
        return temp_row[0]


def is_carbohydrate(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 8]:
            return False
    return True


def morgan_to_list(molecule):
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048).ToBitString()
    return [int(x) for x in fp]


def maccs_to_list(molecule):
    fp = MACCSkeys.GenMACCSKeys(molecule).ToBitString()
    return [int(x) for x in fp]


def maccs_to_structure(maccs_list: list):
    smarts = ''
    idx = [i for i in range(len(maccs_list)) if maccs_list[i] == 1]
    for i in idx:
        smarts = smarts + MACCSkeys.smartsPatts[i][0]
    return smarts
