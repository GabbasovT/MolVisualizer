from rdkit.Chem import Descriptors, MolFromSmiles, rdFingerprintGenerator as fp
import pandas as pd


def calc_fingerprints(mols, radius=3, fp_size=2048):
    """
    генерация молекулярных отпечатков по методу Моргана
    """
    morgan_fpgenerator = fp.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return pd.DataFrame([morgan_fpgenerator.GetFingerprintAsNumPy(m) for m in mols])


PhisChemDescriptors = {"MR": Descriptors.MolMR,
                       "TPSA": Descriptors.TPSA,
                       "num_of_heavy_atoms": Descriptors.HeavyAtomCount,
                       "num_of_atoms": lambda x: x.GetNumAtoms(),
                       "num_valence_electrons": Descriptors.NumValenceElectrons,
                       "num_heteroatoms": Descriptors.NumHeteroatoms,
                       }


# функция для генерации дескрипторов из молекул
def mol_dsc_calc(mols):
    """
    функция для генерации физикохимических дескрипторов
    """
    return pd.DataFrame({k: f(m) for k, f in PhisChemDescriptors.items()}
                        for m in mols)

def number_of_atoms(atom_list, df):
    for i in atom_list:
        df['num_of_{}_atoms'.format(i)] = df['molecules'].apply(lambda x: len(x.GetSubstructMatches(MolFromSmiles(i))))

descriptors_names = list(PhisChemDescriptors.keys()) + ['num_of_{}_atoms'.format(i) for i in ['C', 'O', 'N', 'Cl', 'P', 'Br', 'F']]