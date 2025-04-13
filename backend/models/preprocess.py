from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from rdkit.Chem import MolFromSmiles, rdFingerprintGenerator as fp, AddHs

from chython import smiles
from collections import defaultdict

import pandas as pd
import numpy as np

from .descriptors import *
from .scaler import scaler

from .mol2vec_model import mol2vec_vectors


def standardize(mol_list):
    for m in mol_list:
        try:
            m.clean_stereo()
            m.canonicalize()
        except:
            print(m)


def preprocess_data(smis: list[str]):
    mols = [smiles(m) for m in smis]

    # почистим данные
    standardize(mols)
    # переведём в rdkit
    rdkit_mols = [MolFromSmiles(str(m)) for m in mols]

    rdkit_mols = list(map(lambda x: AddHs(x), rdkit_mols))

    descriptors_transformer = FunctionTransformer(mol_dsc_calc, validate=False)
    morgan_transformer = FunctionTransformer(calc_fingerprints, validate=False)

    rdkit_data = pd.DataFrame(rdkit_mols, columns=['molecules'])

    X_desc = descriptors_transformer.fit_transform(rdkit_data.molecules)
    X_fp = morgan_transformer.fit_transform(rdkit_data.molecules)
    number_of_atoms(['C', 'O', 'N', 'Cl', 'P', 'Br', 'F'], rdkit_data)
    dataset = pd.concat([rdkit_data.drop(columns=['molecules']), X_desc, X_fp],
                                                   axis=1)
    dataset[descriptors_names] = pd.DataFrame(scaler.transform(dataset[descriptors_names]), columns=descriptors_names)

    mol2vec_embs = np.array(mol2vec_vectors(rdkit_data))

    X = np.hstack([mol2vec_embs, np.array(dataset)])
    return X



