import pathlib

import deepsmiles
import pandas as pd
import selfies as sf
import tqdm
from rdkit import Chem

QM9_DIR = pathlib.Path(__file__).parent

constraints = sf.get_semantic_constraints()
constraints['N'] = 6
sf.set_semantic_constraints(constraints)

converter = deepsmiles.Converter(rings=True, branches=True)


def main():
    with open(QM9_DIR / 'raw_qm9.txt', 'r') as f:
        lines = list(map(lambda s: s.rstrip('\n'), f.readlines()))

    columns = ['smiles', 'deep_smiles', 'selfies', 'num_heavy_atoms']
    qm9_rows = []

    for smiles in tqdm.tqdm(lines, desc="Reading SMILES"):

        mol = Chem.MolFromSmiles(smiles)
        num_heavy_atoms = mol.GetNumHeavyAtoms()

        deep_smiles = converter.encode(smiles)
        selfies = sf.encoder(smiles)
        qm9_rows.append((smiles, deep_smiles, selfies, num_heavy_atoms))

        # sanity check
        try:
            converter.decode(deep_smiles)
        except deepsmiles.DecodeError as e:
            print("DecodeError! Error message was '%s'" % e.message)

    qm9_df = pd.DataFrame(qm9_rows, columns=columns)
    qm9_df.to_csv(QM9_DIR / 'qm9.csv', index=False)


if __name__ == '__main__':
    main()
