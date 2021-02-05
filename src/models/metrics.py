import editdistance
import torch
from pytorch_lightning.metrics import Metric

from rdkit.Chem import AllChem
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity


class Accuracy(Metric):

    def __init__(self, ignore_index):
        super().__init__()

        self.ignore_index = ignore_index
        self.add_state('correct', default=torch.tensor(0))
        self.add_state('total', default=torch.tensor(0))

    def update(self, preds, targets):
        assert preds.size() == targets.size()
        assert len(preds.size()) == 2

        compare = 1 - torch.abs(preds - targets).clamp(0, 1)
        compare = torch.where(preds != self.ignore_index, compare, 0)

        self.correct += compare.sum()
        self.total += torch.where(preds != self.ignore_index, 1, 0).sum()

    def compute(self):
        return self.correct.float() / self.total


class EditDistance(Metric):

    def __init__(self, eos_idx):
        super().__init__()

        self.eos_idx = eos_idx
        self.add_state('cul_edit_dist', default=torch.tensor(0))
        self.add_state('total', default=torch.tensor(0))

    def update(self, preds, targets):
        assert preds.size() == targets.size()
        assert len(preds.size()) == 2

        for i in range(preds.size()[0]):
            p = preds[0].tolist()
            t = targets[0].tolist()

            # kind of hacky
            s_p = self._label_to_str(p)
            s_t = self._label_to_str(t)
            dist = editdistance.eval(s_p, s_t)

            self.cul_edit_dist += dist
            self.total += 1

    def compute(self):
        return self.cul_edit_dist.float() / self.total

    def _label_to_str(self, x):
        s = ""
        for i in x:
            assert 0 <= i <= 50

            if i == self.eos_idx:
                break
            else:
                s += chr(i + 48)
        return s


class ChemicalValidity(Metric):

    def __init__(self):
        super().__init__()

        self.add_state('num_valid', default=torch.tensor(0))
        self.add_state('total', default=torch.tensor(0))

    def update(self, mols):
        self.num_valid += (len(mols) - mols.count(None))
        self.total += len(mols)

    def compute(self):
        return self.num_valid.float() / self.total


class ChemicalSimilarity(Metric):

    def __init__(self):
        super().__init__()

        self.add_state('cul_sim', default=torch.tensor(0.0))
        self.add_state('total', default=torch.tensor(0))

    def update(self, mols_a, mols_b):
        assert len(mols_a) == len(mols_b)

        for a, b in zip(mols_a, mols_b):
            if (a is None) or (b is None):
                continue

            fp_a = AllChem.GetMorganFingerprint(a, 2)
            fp_b = AllChem.GetMorganFingerprint(b, 2)
            self.cul_sim += TanimotoSimilarity(fp_a, fp_b)
            self.total += 1

    def compute(self):
        return self.cul_sim.float() / self.total
