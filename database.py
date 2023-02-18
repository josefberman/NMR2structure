from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import MACCSkeys, AllChem
import pandas as pd

def import_nmrshiftdb2_database():
    """
    Loads a SD file containing molecular properties (including NMR spectra)
    :return: List of the molecules from the SD file
    """
    mols = [Chem.AddHs(x) for x in Chem.SDMolSupplier('nmrshiftdb2withsignals.sd') if x is not None]
    return [mol for mol in mols if is_carbohydrate(mol)]


def initialize_dataframe(molecule_list):
    """
    Constructs a pandas DataFrame from a list of molecules (defined by RDKit)
    :param molecule_list: List of RDKit defined molecules
    :return: nmr_df: Dataframe of all molecular properties
    """
    nmr_df = pd.DataFrame([x.GetPropsAsDict() for x in molecule_list if x is not None])
    nmr_df['Name'] = [x.GetProp('_Name') for x in molecule_list if x is not None]
    nmr_df['Smiles'] = [Chem.MolToSmiles(x) for x in molecule_list if x is not None]
    nmr_df['Morgan'] = [morgan_to_list(x) for x in molecule_list if x is not None]
    nmr_df['MACCS'] = [MACCSkeys.GenMACCSKeys(x) for x in molecule_list if x is not None]
    return nmr_df


def import_database_as_df():
    RDLogger.DisableLog('rdApp.*')
    nmr_df = initialize_dataframe(import_nmrshiftdb2_database())
    nmr_df = extract_proton_carbon_spectra(nmr_df)
    return nmr_df


def is_carbohydrate(mol):
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() not in [1, 6, 8]:
            return False
    return True


def extract_proton_carbon_spectra(nmr_df):
    nmr_df['Spectrum 13C'] = nmr_df['Spectrum 13C 0']
    return nmr_df


def morgan_to_list(molecule):
    fp_list = []
    fp = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048).ToBitString()
    for i in fp:
        fp_list.append(int(i))
    return fp_list
