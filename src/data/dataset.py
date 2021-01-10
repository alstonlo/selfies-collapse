import selfies as sf
import torch
from torch.utils import data


class Vocab:

    def __init__(self, alphabet, bos_token, eos_token, pad_token,
                 tokenize_fn):
        alphabet = list(alphabet)
        alphabet.insert(0, pad_token)
        alphabet.insert(1, bos_token)
        alphabet.insert(2, eos_token)

        self.stoi = {c: i for i, c in enumerate(alphabet)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token

        self.bos_idx = self.stoi[bos_token]
        self.eos_idx = self.stoi[eos_token]
        self.pad_idx = self.stoi[pad_token]

        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.stoi)

    def label_encode(self, s):
        x = [self.stoi[c] for c in self.tokenize_fn(s)]
        x.insert(0, self.bos_idx)
        x.append(self.eos_idx)
        return torch.tensor(x, dtype=torch.long)


class LineDataset(data.Dataset):

    def __init__(self, lines, vocab):
        self.vocab = vocab
        self.dataset = list(map(self.vocab.label_encode, lines))
        self.max_len = max(x.size(0) for x in self.dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]


class SMILESDataset(LineDataset):

    def __init__(self, lines):
        alphabet = set()
        for s in lines:
            alphabet.update(set(s))

        vocab = Vocab(
            alphabet=alphabet,
            bos_token='[BOS]',
            eos_token='[EOS]',
            pad_token='[PAD]',
            tokenize_fn=list
        )
        super().__init__(lines, vocab)


class SELFIESDataset(LineDataset):

    def __init__(self, lines):
        alphabet = sf.get_alphabet_from_selfies(lines)

        vocab = Vocab(
            alphabet=alphabet,
            bos_token='[bos]',
            eos_token='[eos]',
            pad_token='[nop]',
            tokenize_fn=self._split_line
        )
        super().__init__(lines, vocab)

    @staticmethod
    def _split_line(s):
        return list(sf.split_selfies(s))
