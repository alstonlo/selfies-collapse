import selfies as sf
import torch
from torch.utils import data


class LineDataset(data.Dataset):

    def __init__(self, lines, split_line,
                 alphabet, sos_token, pad_token):
        self.sos_token = sos_token
        self.pad_token = pad_token

        alphabet = list(sorted(alphabet))
        alphabet.insert(0, pad_token)
        alphabet.insert(1, sos_token)

        self.stoi = {c: i for i, c in enumerate(alphabet)}
        self.itos = {i: c for c, i in self.stoi.items()}

        self.dataset = []
        for s in lines:
            self.dataset.append(
                label_encoded(split_line(s), self.stoi)
            )
        self.max_len = max(x.size(0) for x in self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def get_sos_idx(self):
        return self.stoi[self.sos_token]

    def get_pad_idx(self):
        return self.stoi[self.pad_token]


class SMILESDataset(LineDataset):

    def __init__(self, lines):
        alphabet = set()
        for s in lines:
            alphabet.update(set(s))
        super().__init__(
            lines, list,
            alphabet, sos_token='[SOS]', pad_token='[PAD]'
        )


class SELFIESDataset(LineDataset):

    def __init__(self, lines):
        alphabet = sf.get_alphabet_from_selfies(lines)
        super().__init__(
            lines, self._split_line,
            alphabet, sos_token='[sos]', pad_token='[nop]'
        )

    @staticmethod
    def _split_line(s):
        return list(sf.split_selfies(s))


def label_encoded(symbols, stoi):
    x = [stoi[c] for c in symbols]
    return torch.tensor(x, dtype=torch.long)
