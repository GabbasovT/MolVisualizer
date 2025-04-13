from mol2vec.features import mol2alt_sentence, mol2sentence, MolSentence, DfVec, sentences2vec
from mol2vec.helpers import plot_2D_vectors

from gensim.models import word2vec
w2vec_model = word2vec.Word2Vec.load('model_300dim.pkl')

def mol2vec_vectors(rdkit_smiles):
    mol2vec_data = rdkit_smiles.apply(lambda x: MolSentence(mol2alt_sentence(x['molecules'], 3)), axis=1)
    return [DfVec(x).vec for x in sentences2vec(mol2vec_data, w2vec_model, unseen='UNK')]

