from rdkit import Chem
from rdkit.Chem import MACCSkeys, Draw
from rdkit.Chem.rdchem import Mol
import tensorflow as tf
from keras.models import load_model
from model import encode_spectrum
import numpy as np


def smiles_to_molecule(mol_smiles: str):
    return Chem.MolFromSmiles(mol_smiles)


def draw_molecule(mol: Mol):
    return Draw.MolToImage(mol)


def nmr_to_maccs(nmr_list: list):
    latent_input = encode_spectrum(np.array(nmr_list))
    model = load_model('saved_model/')
    model.predict()



