import pathlib

import selfies as sf
import tqdm
from rdkit import Chem
import pandas as pd

QM9_DIR = pathlib.Path(__file__).parent

constraints = sf.get_semantic_constraints()
constraints['N'] = 6
sf.set_semantic_constraints(constraints)


def mol_to_smiles(mol) -> str:
    return Chem.MolToSmiles(mol, kekuleSmiles=True, canonical=True)


def main():
    with open(QM9_DIR / 'raw_qm9.txt', 'r') as f:
        lines = list(map(lambda s: s.rstrip('\n'), f.readlines()))

    columns = ['SMILES', 'SELFIES', 'num_heavy_atoms']
    qm9_rows = []

    for smiles in tqdm.tqdm(lines, desc="Reading SMILES"):

        mol = Chem.MolFromSmiles(smiles)
        num_heavy_atoms = mol.GetNumHeavyAtoms()

        selfies = sf.encoder(smiles)
        qm9_rows.append((smiles, selfies, num_heavy_atoms))

    qm9_df = pd.DataFrame(qm9_rows, columns=columns)
    qm9_df.to_csv(QM9_DIR / 'qm9.csv', index=False)


if __name__ == '__main__':
    main()
